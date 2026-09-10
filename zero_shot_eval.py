from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from s3dis_sam3d import MP3D_BIMDataset
from s3dis_sam3d.mde import (
    DA3_PROCESS_RES,
    DA3Predictor,
    DepthMetricAccumulator,
    da3_processed_geometry,
)

PROJECT_ROOT = Path(__file__).resolve().parent

# Frozen three-rule benchmark used by the previous PriorBIMDA experiments.
EXPECTED_SELECTION = {
    "hxp": (624, "e6639e7bd16eb7b666a6f22f41ee17ec50a2ce8d8b841427ad489126013bd18b"),
    "759": (518, "d76fd7ed7b5d9b8a6bf882004c990b214e75877349357851d84e4a1b1a01c21e"),
    "1px": (793, "91e589da309b4c3eca24e88667e8c31ceab29fee9a151698d41977beea41e13d"),
}
DEFAULT_SCENES = tuple(EXPECTED_SELECTION)
DEFAULT_DA3_CACHE = PROJECT_ROOT / "outputs/zero_shot_raw/da3_cache"


def frame_set_sha256(frame_ids):
    return hashlib.sha256("\n".join(sorted(frame_ids)).encode()).hexdigest()


def predict(model, sample, device, da3_predictor):
    frame = sample["frame"]
    bim_depth = sample["bim_depth"]
    height, width = frame.image_shape
    process_shape = bim_depth.shape
    da3_depth = da3_predictor.predict_frame(frame)
    rgb = cv2.resize(
        sample["rgb"],
        (process_shape[1], process_shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )

    rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(device)
    da3 = torch.from_numpy(da3_depth)[None, None].to(device)
    bim = torch.from_numpy(bim_depth)[None, None].to(device)
    bim_valid = bim > 0

    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ),
    ):
        output = model(rgb, da3, bim, bim_valid)

    predictions = {
        "da3": da3_depth,
        "global_scale": output["scaled_depth"].float().squeeze().cpu().numpy(),
        "final": output["depth"].float().squeeze().cpu().numpy(),
    }
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
    totals = {key: DepthMetricAccumulator() for key in ("da3", "global_scale", "final")}
    selected_ids = []

    for sample in dataset.iter_scene(name, size=process_shape, progress=True):
        gt_depth = sample["gt_depth"]
        gt_valid = np.isfinite(gt_depth) & (gt_depth > 0)
        selected_ids.append(sample["frame_id"])
        predictions = predict(model, sample, device, da3_predictor)
        for key, prediction in predictions.items():
            totals[key].update(prediction, gt_depth, gt_valid)

    print(f"{name}: {len(frames)} frames, selected={len(selected_ids)}", flush=True)

    selection_hash = frame_set_sha256(selected_ids)
    if name in EXPECTED_SELECTION:
        expected_count, expected_hash = EXPECTED_SELECTION[name]
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
):
    """Evaluate a loaded model on the frozen Matterport3D scenes."""
    scenes = list(scenes)
    dataset = MP3D_BIMDataset(default_mesh_source="obj_wall_filled")
    da3_predictor = DA3Predictor(
        device=device,
        cache_root=da3_cache,
        local_files_only=not allow_network,
    )
    overall = {
        key: DepthMetricAccumulator() for key in ("da3", "global_scale", "final")
    }
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
        for key, overall_values in overall.items():
            overall_values.merge(totals[key])

    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "protocol": {
            "scenes": scenes,
            "process_resolution": DA3_PROCESS_RES,
            "mesh": "registered wall-filled BIMNet OBJ",
            "selection": "GT>10%, BIM hits>20%, camera inside BIM AABB",
            "aggregation": "pixel-micro and frame-macro over selected frames",
            "gt_usage": "scoring and frame selection only",
        },
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
    from mymodel import PriorBIMDA

    parser = argparse.ArgumentParser(
        description="Zero-shot Matterport3D/BIMNet evaluation"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/adapter_3resblocks_raw/best.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/zero_shot_raw/best.json",
    )
    parser.add_argument(
        "--da3-cache",
        type=Path,
        default=DEFAULT_DA3_CACHE,
    )
    parser.add_argument("--scenes", nargs="+", default=list(DEFAULT_SCENES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

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
    )


if __name__ == "__main__":
    main()
