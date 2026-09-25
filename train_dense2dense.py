from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from data import S23PriorBIMDataset
from eval import evaluate, move_to
from model.dense2dense import PriorBIMDA
from train import (
    configure_full_deterministic_model,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    seed_worker,
)
from zero_shot_eval import evaluate_zero_shot


def build_loaders(args, sampler_generator, worker_generator):
    train_set = S23PriorBIMDataset(
        args.dataset_root,
        args.s23_root,
        "train",
        extra_dataset_roots=args.extra_dataset_root,
        extra_dataset_stride=args.extra_dataset_stride,
    )
    val_set = S23PriorBIMDataset(args.dataset_root, args.s23_root, "val", augment=False)
    test_set = S23PriorBIMDataset(
        args.dataset_root, args.s23_root, "test", augment=False
    )
    print(
        f"dataset: train={len(train_set)} {train_set.source_counts}, "
        f"val={len(val_set)}, test={len(test_set)}",
        flush=True,
    )

    counts = Counter(record["region"] for record in train_set.records)
    weights = [counts[record["region"]] ** -0.5 for record in train_set.records]
    sampler = WeightedRandomSampler(
        weights, len(weights), replacement=True, generator=sampler_generator
    )
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(
        train_set,
        sampler=sampler,
        generator=worker_generator,
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader, test_loader


def evaluate_test(model, loader, device, amp, checkpoint_path, output_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"Evaluating test split with {checkpoint_path}", flush=True)
    metrics = evaluate(
        model,
        tqdm(loader, desc="test", unit="batch", dynamic_ncols=True),
        device,
        amp,
    )
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "epoch": checkpoint["epoch"],
        "validation_abs_rel": checkpoint["best_validation_abs_rel"],
        "metrics": metrics,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print("test " + json.dumps(result), flush=True)


def main(model_class=PriorBIMDA):
    parser = argparse.ArgumentParser(
        description="Train the PriorDA-style dense-to-dense log-scale model."
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
    parser.add_argument("--output", default="outputs/dense2dense_syncbim_stride1")
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
        args, sampler_generator, worker_generator
    )

    model = model_class.from_pretrained(local_files_only=args.local_files_only).to(
        device
    )
    if args.full_deterministic:
        configure_full_deterministic_model(model)
    optimizer = torch.optim.AdamW(model.parameter_groups(), weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_loader) / args.accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * steps_per_epoch
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024)
    history = []
    best = float("inf")
    start_epoch = 1
    if args.resume:
        epoch, best, history = load_checkpoint(
            output_dir / "latest.pt", model, optimizer, scheduler, scaler
        )
        start_epoch = epoch + 1

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = (args.seed + (epoch - 1) * 1_000_003) % 2**32
        seed_everything(epoch_seed, full_deterministic=args.full_deterministic)
        sampler_generator.manual_seed(epoch_seed)
        worker_generator.manual_seed((epoch_seed + 1) % 2**32)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
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
                device_type=device.type, dtype=torch.float16, enabled=amp
            ):
                output = model(
                    batch["rgb"],
                    batch["da3_depth"],
                    batch["bim_depth"],
                    batch["bim_valid"],
                )
                losses = model.compute_loss(output, batch)
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
            running += float(losses["total"].detach()) * batch_size
            samples += batch_size
            progress.set_postfix(
                loss=f"{running / samples:.5f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        metrics = evaluate(
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
        score = metrics["final"]["abs_rel"]
        row = {
            "epoch": epoch,
            "train_loss": running / samples,
            "val_abs_rel": score,
            "val_rmse": metrics["final"]["rmse"],
            "val_delta1": metrics["final"]["delta1"],
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
            json.dumps(history, indent=2), encoding="utf-8"
        )

    best_path = output_dir / "best.pt"
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
