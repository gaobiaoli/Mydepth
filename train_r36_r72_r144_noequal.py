from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
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
from train_r36_r72_r144 import (
    evaluate_multiscale,
    evaluate_test,
    evaluate_zero_shot,
)


def main(model_class=PriorBIMDA):
    parser = argparse.ArgumentParser(
        description="Train the R36/R72/R144 model without equivariance loss."
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
        "--output",
        default="outputs/adapter_r36_r72_r144_noequal_syncbim_stride1",
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

    model = model_class.from_pretrained(local_files_only=args.local_files_only).to(
        device
    )
    if args.full_deterministic:
        configure_full_deterministic_model(model)
    print(
        "Equivariance loss disabled; skipping the perturbed-depth forward",
        flush=True,
    )

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
