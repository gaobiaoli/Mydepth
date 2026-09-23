from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from eval import evaluate, move_to
from model.mymodel import PriorBIMDA
from train import (
    build_loaders,
    configure_full_deterministic_model,
    evaluate_test,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
)
from zero_shot_eval import evaluate_zero_shot


def main(model_class=PriorBIMDA):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", default="/mnt/priorbimda-data/area1_priorbimda_504"
    )
    parser.add_argument(
        "--extra-dataset-root",
        action="append",
        default=[],
        help="train-only SyncBIM root; repeat this option to use multiple roots",
    )
    parser.add_argument(
        "--extra-dataset-stride",
        type=int,
        default=1,
        help="keep every Nth record from each extra training dataset",
    )
    parser.add_argument("--s23-root", default="/home/bgao491/Stanford2D3DS/no_xyz")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", default="outputs/raw4")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--full-deterministic",
        action="store_true",
        help=(
            "enable strict deterministic algorithms and replace DINOv2 "
            "bicubic position interpolation with deterministic matmuls"
        ),
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--zero-shot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run zero-shot evaluation after the Area1 test split",
    )
    args = parser.parse_args()

    seed_everything(
        args.seed,
        full_deterministic=args.full_deterministic,
    )
    device = torch.device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    sampler_generator = torch.Generator().manual_seed(args.seed)
    train_worker_generator = torch.Generator().manual_seed(args.seed + 1)
    train_loader, val_loader, test_loader = build_loaders(
        args,
        sampler_generator,
        train_worker_generator,
    )

    model = model_class.from_pretrained(local_files_only=args.local_files_only)
    model = model.to(device)
    if args.full_deterministic:
        configure_full_deterministic_model(model)
        print(
            "Full deterministic mode: strict algorithms and deterministic "
            "DINOv2 position interpolation; position embeddings trainable",
            flush=True,
        )
    print(
        "Equivariance loss disabled; skipping the perturbed-depth forward",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameter_groups(factor=1.0), weight_decay=0.01
    )
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
        print(f"Resuming from epoch {epoch} with best validation abs_rel {best:.5f}")
        start_epoch = epoch + 1

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_seed = (args.seed + (epoch - 1) * 1_000_003) % 2**32
        seed_everything(
            epoch_seed,
            full_deterministic=args.full_deterministic,
        )
        sampler_generator.manual_seed(epoch_seed)
        train_worker_generator.manual_seed((epoch_seed + 1) % 2**32)
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
    if not best_path.is_file():
        raise FileNotFoundError(f"Best checkpoint not found: {best_path}")
    del optimizer, scheduler, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    evaluate_test(model, test_loader, device, amp, best_path, output_dir)
    if args.zero_shot:
        print("Evaluating the default zero-shot scenes", flush=True)
        evaluate_zero_shot(
            model,
            device,
            best_path,
            output_dir / "zero_shot_metrics.json",
            allow_network=not args.local_files_only,
        )


if __name__ == "__main__":
    main()
