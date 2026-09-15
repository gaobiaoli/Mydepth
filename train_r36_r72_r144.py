from __future__ import annotations

import argparse
import hashlib
import json
import math
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
from torch.nn import functional as F
from tqdm import tqdm

from eval import move_to
from model.mymodel_r36_r72_r144 import PriorBIMDA
from train import (
    build_loaders,
    configure_full_deterministic_model,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)
from zero_shot_eval import DEFAULT_DA3_CACHE, DEFAULT_SCENES, EXPECTED_SELECTION

STAGE_NAMES = ("da3", "global_scale", "r36", "r72", "r144")
PREDICTION_NAMES = (*STAGE_NAMES, "final")


def stage_predictions(output, da3_depth):
    """Build the cumulative depth prediction after each residual stage."""
    size = da3_depth.shape[-2:]

    def resize(value):
        return F.interpolate(
            value.float(),
            size=size,
            mode="bilinear",
            align_corners=False,
        )

    residual36 = resize(output["log_residual_r36"])
    residual72 = resize(output["log_residual_r72"])
    scaled_depth = output["scaled_depth"].float()
    depth36 = (scaled_depth * residual36.exp()).clamp(1e-3, 128)
    depth72 = (scaled_depth * (residual36 + residual72).exp()).clamp(1e-3, 128)
    final = output["depth"].float()
    return {
        "da3": da3_depth.float(),
        "global_scale": scaled_depth,
        "r36": depth36,
        "r72": depth72,
        "r144": final,
    }


@torch.inference_mode()
def evaluate_multiscale(model, loader, device, amp=True):
    """Evaluate DA3, scale and all three cumulative residual stages."""
    totals = {name: DepthMetricAccumulator() for name in STAGE_NAMES}
    model.eval()
    for batch in loader:
        batch = move_to(batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            output = model(
                batch["rgb"],
                batch["da3_depth"],
                batch["bim_depth"],
                batch["bim_valid"],
            )
        predictions = {
            name: value[:, 0].float().cpu().numpy()
            for name, value in stage_predictions(output, batch["da3_depth"]).items()
        }
        targets = batch["gt_depth"][:, 0].float().cpu().numpy()
        valid = batch["gt_valid"][:, 0].bool().cpu().numpy()
        for index in range(targets.shape[0]):
            for name in STAGE_NAMES:
                totals[name].update(
                    predictions[name][index],
                    targets[index],
                    valid[index],
                )
    metrics = {name: total.compute() for name, total in totals.items()}
    metrics["final"] = metrics["r144"]
    return metrics


def evaluate_test(model, loader, device, amp, checkpoint_path, output_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"Evaluating test split with {checkpoint_path}", flush=True)
    metrics = evaluate_multiscale(
        model,
        tqdm(loader, desc="test", unit="batch", dynamic_ncols=True),
        device,
        amp,
    )
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "epoch": checkpoint["epoch"],
        "validation_abs_rel": checkpoint["best_validation_abs_rel"],
        "protocol": {
            "predictions": list(PREDICTION_NAMES),
            "final_alias": "r144",
            "aggregation": "pixel-micro and frame-macro",
        },
        "metrics": metrics,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print("test " + json.dumps(result), flush=True)
    return result


def frame_set_sha256(frame_ids):
    return hashlib.sha256("\n".join(sorted(frame_ids)).encode()).hexdigest()


def predict_zero_shot(model, sample, device, da3_predictor):
    frame = sample["frame"]
    bim_depth = sample["bim_depth"]
    height, width = frame.image_shape
    da3_depth = da3_predictor.predict_frame(frame)

    rgb = cv2.imread(str(frame.rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Cannot read RGB image: {frame.rgb_path}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    rgb = (
        cv2.resize(
            rgb,
            (bim_depth.shape[1], bim_depth.shape[0]),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        / 255.0
    )
    rgb = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].to(device)
    da3 = torch.from_numpy(da3_depth)[None, None].to(device)
    bim = torch.from_numpy(bim_depth)[None, None].to(device)
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
        output = model(rgb, da3, bim, bim > 0)
        predictions = stage_predictions(output, da3)
    return {
        name: cv2.resize(
            value.float().squeeze().cpu().numpy(),
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        for name, value in predictions.items()
    }


def evaluate_zero_shot_scene(name, model, device, dataset, da3_predictor):
    frames = dataset.frames(name)
    process_shapes = {
        da3_processed_geometry(frame.image_shape, frame.intrinsics)[:2]
        for frame in frames
    }
    if len(process_shapes) != 1:
        raise ValueError(f"{name} has inconsistent DA3 input shapes: {process_shapes}")
    process_shape = process_shapes.pop()
    totals = {key: DepthMetricAccumulator() for key in STAGE_NAMES}
    selected_ids = []

    for sample in dataset.iter_scene(name, size=process_shape, progress=True):
        target = sample["gt_depth"]
        valid = np.isfinite(target) & (target > 0)
        selected_ids.append(sample["frame_id"])
        predictions = predict_zero_shot(model, sample, device, da3_predictor)
        for key, prediction in predictions.items():
            totals[key].update(prediction, target, valid)

    selection_hash = frame_set_sha256(selected_ids)
    if name in EXPECTED_SELECTION:
        expected_count, expected_hash = EXPECTED_SELECTION[name]
        if (len(selected_ids), selection_hash) != (expected_count, expected_hash):
            raise RuntimeError(
                f"{name} selection changed: got {len(selected_ids)} / {selection_hash}"
            )
    print(f"{name}: {len(frames)} frames, selected={len(selected_ids)}", flush=True)
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
    """Run the frozen zero-shot protocol with all residual stages exposed."""
    scenes = list(scenes)
    dataset = MP3D_BIMDataset(default_mesh_source="obj_wall_filled")
    da3_predictor = DA3Predictor(
        device=device,
        cache_root=da3_cache,
        local_files_only=not allow_network,
    )
    overall = {key: DepthMetricAccumulator() for key in STAGE_NAMES}
    scene_results = {}
    model.eval()
    for name in scenes:
        result, totals = evaluate_zero_shot_scene(
            name,
            model,
            device,
            dataset,
            da3_predictor,
        )
        scene_results[name] = result
        result["metrics"]["final"] = result["metrics"]["r144"]
        for key in STAGE_NAMES:
            overall[key].merge(totals[key])

    overall_metrics = {key: value.compute() for key, value in overall.items()}
    overall_metrics["final"] = overall_metrics["r144"]
    summary = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "protocol": {
            "scenes": scenes,
            "process_resolution": DA3_PROCESS_RES,
            "mesh": "registered wall-filled BIMNet OBJ",
            "selection": "GT>10%, BIM hits>20%, camera inside BIM AABB",
            "aggregation": "pixel-micro and frame-macro over selected frames",
            "predictions": list(PREDICTION_NAMES),
            "final_alias": "r144",
            "gt_usage": "scoring and frame selection only",
        },
        "scenes": scene_results,
        "overall": overall_metrics,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    print(f"Saved {output.resolve()}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Train the adapter + F36/F72/F144 residual model."
    )
    parser.add_argument(
        "--dataset-root", default="/mnt/priorbimda-data/area1_priorbimda_504"
    )
    parser.add_argument(
        "--extra-dataset-root",
        action="append",
        default=[],
        help="train-only SyncBIM root; repeat to use multiple roots",
    )
    parser.add_argument("--extra-dataset-stride", type=int, default=1)
    parser.add_argument("--s23-root", default="/home/bgao491/Stanford2D3DS/no_xyz")
    parser.add_argument(
        "--output", default="outputs/adapter_r36_r72_r144_syncbim_stride1"
    )
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full-deterministic", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--zero-shot",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    seed_everything(args.seed, full_deterministic=args.full_deterministic)
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler_generator = torch.Generator().manual_seed(args.seed)
    worker_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader, val_loader, test_loader = build_loaders(
        args,
        sampler_generator,
        worker_generator,
    )

    model = PriorBIMDA.from_pretrained(local_files_only=args.local_files_only).to(
        device
    )
    if args.full_deterministic:
        configure_full_deterministic_model(model)
    optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * steps_per_epoch,
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    history = []
    best = float("inf")
    start_epoch = 1
    if args.resume:
        epoch, best, history = load_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
        )
        start_epoch = epoch + 1

    loss_names = (
        "total",
        "depth",
        "depth_r36",
        "depth_r72",
        "depth_r144",
        "scale",
        "equivariance",
    )
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = (args.seed + (epoch - 1) * 1_000_003) % 2**32
        seed_everything(epoch_seed, full_deterministic=args.full_deterministic)
        sampler_generator.manual_seed(epoch_seed)
        worker_generator.manual_seed((epoch_seed + 1) % 2**32)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = {name: 0.0 for name in loss_names}
        samples = 0

        progress = tqdm(
            train_loader,
            desc=f"train {epoch}/{args.epochs}",
            unit="batch",
            dynamic_ncols=True,
        )
        for step, batch in enumerate(progress, start=1):
            batch = move_to(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ):
                output = model(
                    batch["rgb"],
                    batch["da3_depth"],
                    batch["bim_depth"],
                    batch["bim_valid"],
                )
                selected = (
                    torch.rand((batch["rgb"].shape[0], 1, 1, 1), device=device) < 0.5
                )
                log_factor = torch.empty_like(output["log_scale"]).uniform_(
                    -0.2,
                    0.2,
                )
                log_factor = torch.where(
                    selected,
                    log_factor,
                    torch.zeros_like(log_factor),
                )
                changed_scale = model.predict_log_scale(
                    batch["rgb"],
                    batch["da3_depth"] * log_factor.exp(),
                    batch["bim_depth"],
                    batch["bim_valid"],
                )
                equivariance = changed_scale + log_factor - output["log_scale"]
                losses = model.compute_loss(output, batch, equivariance)

            scaler.scale(losses["total"] / args.accumulation).backward()
            if step % args.accumulation == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            batch_size = batch["rgb"].shape[0]
            samples += batch_size
            for name in loss_names:
                running[name] += float(losses[name].detach()) * batch_size
            progress.set_postfix(
                loss=f"{running['total'] / samples:.5f}",
                r36=f"{running['depth_r36'] / samples:.4f}",
                r72=f"{running['depth_r72'] / samples:.4f}",
                r144=f"{running['depth_r144'] / samples:.4f}",
            )

        metrics = evaluate_multiscale(
            model,
            tqdm(
                val_loader,
                desc=f"val {epoch}/{args.epochs}",
                unit="batch",
                dynamic_ncols=True,
            ),
            device,
            amp,
        )
        score = metrics["final"]["pixel_micro"]["abs_rel"]
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / samples for name, value in running.items()},
            "val_abs_rel": score,
            "val_frame_abs_rel": metrics["final"]["frame_macro"]["abs_rel"],
            "val_rmse": metrics["final"]["pixel_micro"]["rmse"],
            "val_delta1": metrics["final"]["pixel_micro"]["delta1"],
            "val_r36_abs_rel": metrics["r36"]["pixel_micro"]["abs_rel"],
            "val_r72_abs_rel": metrics["r72"]["pixel_micro"]["abs_rel"],
            "val_r144_abs_rel": metrics["r144"]["pixel_micro"]["abs_rel"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        save_checkpoint(
            output_dir / "latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            min(best, score),
            history,
        )
        if score < best:
            best = score
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best,
                history,
            )
        (output_dir / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    best_path = output_dir / "best.pt"
    if not best_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
    del optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluate_test(model, test_loader, device, amp, best_path, output_dir)
    if args.zero_shot:
        evaluate_zero_shot(
            model,
            device,
            best_path,
            output_dir / "zero_shot_metrics.json",
            allow_network=not args.local_files_only,
        )


if __name__ == "__main__":
    main()
