from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# from MyDepth import loss
from data import S23PriorBIMDataset
from eval import evaluate, move_to
from model.mymodel import PriorBIMDA
from zero_shot_eval import evaluate_zero_shot


def seed_everything(seed, deterministic=True, full_deterministic=False):
    """Seed all RNGs and optionally enable the strict deterministic profile."""
    if full_deterministic:
        deterministic = True
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
            # This is set before the first CUDA operation in main().
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # This affects child processes. For the main interpreter, set the same
        # variable before launching Python (train.sh already does this).
        os.environ.setdefault("PYTHONHASHSEED", str(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuDNN
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # The default profile preserves the previous warn-only behaviour. The
        # full profile fails immediately if a nondeterministic kernel remains.
        torch.use_deterministic_algorithms(
            True,
            # warn_only=not full_deterministic,
            warn_only=not full_deterministic,
        )

        if full_deterministic:
            pass
            # torch.set_deterministic_debug_mode("error")
            # torch.backends.cuda.matmul.allow_tf32 = False
            # torch.backends.cudnn.allow_tf32 = False
            # torch.set_float32_matmul_precision("highest")
            # torch.backends.cuda.enable_flash_sdp(False)
            # torch.backends.cuda.enable_mem_efficient_sdp(False)
            # torch.backends.cuda.enable_math_sdp(True)
            # if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            #     torch.backends.cuda.enable_cudnn_sdp(False)
            # if hasattr(
            #     torch.backends.cuda.matmul,
            #     "allow_fp16_reduced_precision_reduction",
            # ):
            #     torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            #         False
            #     )
            # if hasattr(
            #     torch.backends.cuda.matmul,
            #     "allow_bf16_reduced_precision_reduction",
            # ):
            #     torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            #         False
            #     )

    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


def configure_full_deterministic_model(model):
    """Remove the trainable bicubic CUDA backward path in DINOv2."""
    position_embeddings = model.dav2.backbone.embeddings.position_embeddings
    position_embeddings.requires_grad_(False)


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_loaders(args, sampler_generator,train_worker_generator):
    train_set = S23PriorBIMDataset(
        args.dataset_root,
        args.s23_root,
        "train",
    )
    val_set = S23PriorBIMDataset(args.dataset_root, args.s23_root, "val", augment=False)
    test_set = S23PriorBIMDataset(
        args.dataset_root, args.s23_root, "test", augment=False
    )

    # The best run sampled large rooms less often: weight(room) = count^-0.5.
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
    train_loader = DataLoader(train_set, sampler=sampler,generator=train_worker_generator, drop_last=True, **common)
    val_loader = DataLoader(val_set, shuffle=False,generator=torch.Generator().manual_seed(args.seed), drop_last=False, **common)
    test_loader = DataLoader(test_set, shuffle=False,generator=torch.Generator().manual_seed(args.seed), drop_last=False, **common)
    return train_loader, val_loader, test_loader


def save_checkpoint(path, model, optimizer, scheduler, scaler, epoch, best, history):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_validation_abs_rel": best,
            "history": history,
        },
        path,
    )


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    epoch = checkpoint["epoch"]
    best = checkpoint["best_validation_abs_rel"]
    history = checkpoint["history"]
    return epoch, best, history


def evaluate_test(model, loader, device, amp, checkpoint_path, output_dir):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    best_epoch = checkpoint["epoch"]
    validation_abs_rel = checkpoint["best_validation_abs_rel"]
    del checkpoint
    print(f"Evaluating test split with {checkpoint_path}", flush=True)
    metrics = evaluate(
        model,
        tqdm(loader, desc="test", unit="batch", dynamic_ncols=True),
        device,
        amp,
    )
    result = {
        "checkpoint": str(checkpoint_path.resolve()),
        "epoch": best_epoch,
        "validation_abs_rel": validation_abs_rel,
        "metrics": metrics,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    print("test " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", default="/mnt/priorbimda-data/area1_priorbimda_504"
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
            "enable strict deterministic algorithms, Math SDP only, no TF32, "
            "and freeze DINOv2 position embeddings"
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
    # loss_log = []
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


    model = PriorBIMDA.from_pretrained(local_files_only=args.local_files_only)
    if args.full_deterministic:
        configure_full_deterministic_model(model)
        print(
            "Full deterministic mode: strict algorithms, Math SDP only, "
            "TF32 disabled, DINOv2 position embeddings frozen",
            flush=True,
        )
    model = model.to(
        device,
    )
    # model.enable_gradient_checkpointing()
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
        train_worker_generator.manual_seed(
            (epoch_seed + 1) % 2**32
        )
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

                selected = (
                    torch.rand((batch["rgb"].shape[0], 1, 1, 1), device=device) < 0.5
                )
                log_factor = torch.empty_like(output["log_scale"]).uniform_(-0.2, 0.2)
                log_factor = torch.where(
                    selected, log_factor, torch.zeros_like(log_factor)
                )
                changed_scale = model.predict_log_scale(
                    batch["rgb"],
                    batch["da3_depth"] * log_factor.exp(),
                    batch["bim_depth"],
                    batch["bim_valid"],
                )
                equivariance = changed_scale + log_factor - output["log_scale"]
                losses = model.compute_loss(output, batch, equivariance)
                # loss_log.append(losses["total"].detach().cpu().numpy())
                # if step == 100:
                #     np.save(output_dir / "loss_log.npy", np.array(loss_log))
                #     return
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
