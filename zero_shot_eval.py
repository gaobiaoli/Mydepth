from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from s3dis_sam3d import MP3D_BIMDataset
from s3dis_sam3d.mde import (
    DA3_PROCESS_RES,
    DA3Predictor,
    UniDepthV2Predictor,
    MoGe2Predictor,
    MoGe3Predictor,
    DepthMetricAccumulator,
    da3_processed_geometry,
)
# DA3Predictor = MoGe3Predictor
PROJECT_ROOT = Path(__file__).resolve().parent

# Frozen three-rule benchmark used by the previous PriorBIMDA experiments.
EXPECTED_SELECTION = {
    "hxp": (624, "e6639e7bd16eb7b666a6f22f41ee17ec50a2ce8d8b841427ad489126013bd18b"),
    "759": (518, "d76fd7ed7b5d9b8a6bf882004c990b214e75877349357851d84e4a1b1a01c21e"),
    "1px": (793, "91e589da309b4c3eca24e88667e8c31ceab29fee9a151698d41977beea41e13d"),
}
DEFAULT_SCENES = tuple(EXPECTED_SELECTION)
ALL_SCENE = "all"
DEFAULT_DA3_CACHE = "/mnt/priorbimda-data/zero_shot_raw/da3_cache"


def frame_set_sha256(frame_ids):
    return hashlib.sha256("\n".join(sorted(frame_ids)).encode()).hexdigest()


def resolve_scenes(dataset, scenes):
    """Validate explicit scene IDs or expand ``all`` to every usable pair."""
    if isinstance(scenes, str):
        scenes = [scenes]
    scenes = [
        item.strip()
        for value in scenes
        for item in str(value).split(",")
        if item.strip()
    ]
    if not scenes:
        raise ValueError("At least one scene ID is required")

    select_all = [str(scene).casefold() == ALL_SCENE for scene in scenes]
    mp3d_scene_ids = set(dataset.mp3d_dataset.scene_ids)

    def validate(scene):
        if scene.matterport_scan_id not in mp3d_scene_ids:
            raise ValueError(
                f"Matterport3D scan is unavailable for BIMNet scene {scene.key!r}"
            )

    if any(select_all):
        if len(scenes) != 1:
            raise ValueError(f"{ALL_SCENE!r} cannot be combined with explicit scenes")
        resolved = [
            scene.key
            for scene in dataset.bimnet_dataset.scenes
            if scene.matterport_scan_id in mp3d_scene_ids
        ]
        if not resolved:
            raise RuntimeError("No paired Matterport3D/BIMNet scenes were found")
        return resolved

    resolved = []
    seen = set()
    for identifier in scenes:
        floor_scenes = dataset.bimnet_dataset.scenes_for_scan(identifier)
        if not floor_scenes:
            floor_scenes = (dataset.bimnet_dataset.scene(identifier),)

        # A Matterport scan may contain several BIMNet floor scenes. Evaluate
        # each floor separately because MP3D_BIMDataset applies its containment
        # rule to one floor envelope at a time.
        multiple_floors = len(floor_scenes) > 1
        for scene in floor_scenes:
            validate(scene)
            if scene.key in seen:
                continue
            resolved.append(scene.key if multiple_floors else identifier)
            seen.add(scene.key)
    return resolved


def prediction_depths(output, da3_depth):
    """Return the legacy predictions plus every cumulative residual stage."""
    scaled_depth = output["scaled_depth"].float()
    predictions = {
        "da3": da3_depth.float(),
        "global_scale": scaled_depth,
    }

    stages = []
    prefix = "log_residual_r"
    for name, residual in output.items():
        if name.startswith(prefix) and name[len(prefix) :].isdigit():
            stages.append((int(name[len(prefix) :]), residual))

    cumulative = torch.zeros_like(scaled_depth)
    for resolution, residual in sorted(stages):
        residual = F.interpolate(
            residual.float(),
            size=scaled_depth.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cumulative = cumulative + residual
        predictions[f"r{resolution}"] = (
            scaled_depth * cumulative.exp()
        ).clamp(1e-3, 128)

    predictions["final"] = output["depth"].float()
    return predictions


def predict(model, sample, device, da3_predictor):
    frame = sample["frame"]
    bim_depth = sample["bim_depth"]
    height, width = frame.image_shape
    process_shape = bim_depth.shape
    da3_depth = da3_predictor.predict_frame(frame)

    # Match the original PriorBIMDA evaluator exactly: resize uint8 RGB first,
    # then convert it to the [0, 1] float range expected by PriorBIMDA.  The
    # toolkit's ``frame.rgb`` is float32 but deliberately retains [0, 255], so
    # passing sample["rgb"] through unchanged would be saturated by the
    # model's clamp(0, 1).
    rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Cannot read RGB image: {frame.rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(
        rgb,
        (process_shape[1], process_shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    rgb = rgb.astype(np.float32) / 255.0

    rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(device)
    da3 = torch.from_numpy(da3_depth)[None, None].to(device)
    bim = torch.from_numpy(bim_depth)[None, None].to(device)
    bim_valid = bim > 0

    amp_dtype = (
        torch.bfloat16
        if device.type == "cuda" and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ),
    ):
        output = model(rgb, da3, bim, bim_valid)

    predictions = {
        name: value.squeeze().cpu().numpy()
        for name, value in prediction_depths(output, da3).items()
    }
    predictions["da3"] = da3_depth
    return {
        name: cv2.resize(value, (width, height), interpolation=cv2.INTER_LINEAR)
        for name, value in predictions.items()
    }


def evaluate_scene(name, model, device, dataset, da3_predictor):
    frames = dataset.frames(name)
    process_shapes = {
        da3_processed_geometry(frame.image_shape, frame.intrinsics)[:2]
        for frame in frames
    }
    if len(process_shapes) != 1:
        raise ValueError(
            f"{name} contains inconsistent DA3 input shapes: {process_shapes}"
        )
    process_shape = process_shapes.pop()
    totals = {}
    selected_ids = []

    for sample in dataset.iter_scene(name, size=process_shape, progress=True):
        gt_depth = sample["gt_depth"]
        gt_valid = np.isfinite(gt_depth) & (gt_depth > 0)
        selected_ids.append(sample["frame_id"])
        predictions = predict(model, sample, device, da3_predictor)
        if not totals:
            totals = {
                key: DepthMetricAccumulator()
                for key in predictions
            }
        elif predictions.keys() != totals.keys():
            raise RuntimeError(
                f"{name} returned inconsistent prediction stages: "
                f"{list(predictions)} != {list(totals)}"
            )
        for key, prediction in predictions.items():
            totals[key].update(prediction, gt_depth, gt_valid)

    print(f"{name}: {len(frames)} frames, selected={len(selected_ids)}", flush=True)

    selection_hash = frame_set_sha256(selected_ids)
    benchmark_name = dataset.bimnet_dataset.scene(name).scene_id
    if benchmark_name in EXPECTED_SELECTION:
        expected_count, expected_hash = EXPECTED_SELECTION[benchmark_name]
        if (len(selected_ids), selection_hash) != (expected_count, expected_hash):
            raise RuntimeError(
                f"{name} selection changed: got {len(selected_ids)} / {selection_hash}"
            )

    return {
        "source_frames": len(frames),
        "selected_frames": len(selected_ids),
        "selected_frame_ids_sha256": selection_hash,
        "metrics": {key: value.compute() for key, value in totals.items()},
    }, totals


def evaluate_zero_shot(
    model,
    device,
    checkpoint,
    output,
    scenes=DEFAULT_SCENES,
    da3_cache=DEFAULT_DA3_CACHE,
    allow_network=False,
    mesh_source="obj",
):
    """Evaluate a loaded model on selected Matterport3D/BIMNet scene pairs."""
    dataset = MP3D_BIMDataset(default_mesh_source=mesh_source)
    scenes = resolve_scenes(dataset, scenes)
    print(f"Evaluating {len(scenes)} zero-shot scene(s): {', '.join(scenes)}", flush=True)
    da3_predictor = DA3Predictor(
        device=device,
        cache_root=da3_cache,
        local_files_only=not allow_network,
    )
    overall = {}
    scene_results = {}

    model.eval()
    for name in scenes:
        result, totals = evaluate_scene(
            name,
            model,
            device,
            dataset,
            da3_predictor,
        )
        scene_results[name] = result
        if overall and totals.keys() != overall.keys():
            raise RuntimeError(
                f"{name} returned different prediction stages: "
                f"{list(totals)} != {list(overall)}"
            )
        for key, values in totals.items():
            overall.setdefault(key, DepthMetricAccumulator()).merge(values)

    protocol = {
        "scenes": scenes,
        "process_resolution": DA3_PROCESS_RES,
        "mesh": f"registered BIMNet {mesh_source}",
        "selection": "GT>10%, BIM hits>20%, camera inside BIM AABB",
        "aggregation": "pixel-micro and frame-macro over selected frames",
        "gt_usage": "scoring and frame selection only",
    }
    residual_stages = [
        name for name in overall if name.startswith("r") and name[1:].isdigit()
    ]
    if residual_stages:
        protocol["predictions"] = list(overall)
        protocol["final_alias"] = residual_stages[-1]

    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "protocol": protocol,
        "scenes": scene_results,
        "overall": {key: value.compute() for key, value in overall.items()},
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Saved {output.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Matterport3D/BIMNet evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/area1_syncbim_stride1/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/area1_syncbim_stride1/best1.json",
    )
    parser.add_argument(
        "--da3-cache",
        type=Path,
        default=DEFAULT_DA3_CACHE,
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        metavar="SCENE",
        default=list(DEFAULT_SCENES),
        # default=["s9h"],
        help=(
            "one or more BIMNet/MP3D scene IDs (space- or comma-separated), "
            f"or {ALL_SCENE!r} for every usable pair"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument(
        "--mesh-source",
        choices=("obj", "wall-filled"),
        default="obj",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False,
        mmap=True,
    )
    state = checkpoint.get("model", checkpoint)
    multiscale = any(name.startswith("stage72.") for name in state)
    del state, checkpoint
    if multiscale:
        from model.mymodel_r36_r72_r144 import PriorBIMDA

        print("Detected adapter R36/R72/R144 checkpoint", flush=True)
    else:
        from model.mymodel import PriorBIMDA

    model = PriorBIMDA.from_pretrained(local_files_only=not args.allow_network)
    model.load_checkpoint(args.checkpoint)
    model.to(device).eval()

    evaluate_zero_shot(
        model,
        device,
        args.checkpoint,
        args.output,
        scenes=args.scenes,
        da3_cache=args.da3_cache,
        allow_network=args.allow_network,
        mesh_source=(
            "obj_wall_filled" if args.mesh_source == "wall-filled" else "obj"
        ),
    )


if __name__ == "__main__":
    main()
