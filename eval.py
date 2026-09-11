from __future__ import annotations

import argparse
import json

import torch

from data import build_area1_dataloader
from model.baseline import PriorBIMDA


def move_to(batch, device):
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def update_metrics(total, prediction, target, valid):
    support = (valid > 0) & torch.isfinite(target) & (target > 0)
    if not bool((torch.isfinite(prediction[support]) & (prediction[support] > 0)).all()):
        raise FloatingPointError("Invalid prediction on fixed GT support")

    prediction = prediction[support].float()
    target = target[support].float()
    error = prediction - target
    log_error = torch.log(prediction) - torch.log(target)
    ratio = torch.maximum(prediction / target, target / prediction)
    total["count"] += prediction.numel()
    total["abs_rel"] += float((error.abs() / target).sum())
    total["squared_error"] += float(error.square().sum())
    total["absolute_error"] += float(error.abs().sum())
    total["squared_log_error"] += float(log_error.square().sum())
    total["delta1"] += int((ratio < 1.25).sum())
    total["delta2"] += int((ratio < 1.25**2).sum())


def finish_metrics(total):
    count = total["count"]
    return {
        "abs_rel": total["abs_rel"] / count,
        "rmse": (total["squared_error"] / count) ** 0.5,
        "mae": total["absolute_error"] / count,
        "delta1": total["delta1"] / count,
        "delta2": total["delta2"] / count,
        "rmse_log": (total["squared_log_error"] / count) ** 0.5,
        "count": count,
    }


@torch.inference_mode()
def evaluate(model, loader, device, amp=True):
    model.eval()
    names = ("da3", "global_scale", "final")
    totals = {
        name: {
            "count": 0,
            "abs_rel": 0.0,
            "squared_error": 0.0,
            "absolute_error": 0.0,
            "squared_log_error": 0.0,
            "delta1": 0,
            "delta2": 0,
        }
        for name in names
    }

    for batch in loader:
        batch = move_to(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=amp and device.type == "cuda"):
            output = model(
                batch["rgb"], batch["da3_depth"],
                batch["bim_depth"], batch["bim_valid"],
            )
        predictions = {
            "da3": batch["da3_depth"],
            "global_scale": output["scaled_depth"],
            "final": output["depth"],
        }
        for name, prediction in predictions.items():
            update_metrics(
                totals[name], prediction, batch["gt_depth"], batch["gt_valid"]
            )

    return {name: finish_metrics(total) for name, total in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--s23-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    loader = build_area1_dataloader(
        dataset_root=args.dataset_root,
        s23_root=args.s23_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=False,
        drop_last=False,
    )
    model = PriorBIMDA.from_pretrained(local_files_only=args.local_files_only)
    model.load_checkpoint(args.checkpoint)
    model.to(device)
    print(json.dumps(evaluate(model, loader, device), indent=2))


if __name__ == "__main__":
    main()
