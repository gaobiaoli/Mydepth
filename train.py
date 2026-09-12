from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from types import MethodType

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



def configure_full_deterministic_model_trainable(
    model,
    target_height: int = 504,
    target_width: int = 504,
):
    """
    Keep DINOv2 positional embeddings trainable while removing the
    nondeterministic CUDA bicubic-interpolation backward path.

    The pretrained positional embeddings are resized ONCE to the target
    patch grid under no_grad(), then re-registered as a trainable Parameter.

    For 504x504 input with patch_size=14:
        504 / 14 = 36
    so the new positional embedding has shape:
        [1, 1 + 36*36, hidden_dim]

    Important:
    - Call this BEFORE constructing the optimizer.
    - Prefer calling it AFTER model.to(device), so the one-time interpolation
      uses the same device/kernel as normal DINOv2 forward interpolation.
    - Training should remain at this target patch grid. Inference may still
      use other resolutions; DINOv2 can interpolate this learned PE again.
    """
    embeddings = model.dav2.backbone.embeddings
    position_embeddings = embeddings.position_embeddings

    if position_embeddings.ndim != 3 or position_embeddings.shape[0] != 1:
        raise ValueError(
            "Expected DINOv2 position_embeddings with shape [1, N, C], "
            f"got {tuple(position_embeddings.shape)}"
        )

    # Infer patch size directly from the patch projection Conv2d.
    projection = embeddings.patch_embeddings.projection

    patch_size = projection.kernel_size
    if isinstance(patch_size, int):
        patch_h = patch_w = patch_size
    else:
        patch_h, patch_w = patch_size

    if target_height % patch_h != 0 or target_width % patch_w != 0:
        raise ValueError(
            f"Target resolution {(target_height, target_width)} must be "
            f"divisible by patch size {(patch_h, patch_w)}"
        )

    target_grid_h = target_height // patch_h
    target_grid_w = target_width // patch_w

    # Original pretrained positional grid.
    num_source_patches = position_embeddings.shape[1] - 1
    source_grid = math.isqrt(num_source_patches)

    if source_grid * source_grid != num_source_patches:
        raise ValueError(
            "Expected a square pretrained positional grid, but got "
            f"{num_source_patches} patch positions."
        )

    hidden_dim = position_embeddings.shape[-1]
    original_dtype = position_embeddings.dtype
    original_device = position_embeddings.device

    with torch.no_grad():
        # Keep CLS positional embedding unchanged.
        cls_pos = position_embeddings[:, :1].float()

        # [1, N, C] -> [1, C, H0, W0]
        patch_pos = position_embeddings[:, 1:].float()
        patch_pos = patch_pos.reshape(
            1,
            source_grid,
            source_grid,
            hidden_dim,
        ).permute(0, 3, 1, 2)

        # One-time interpolation.
        # No backward graph is created here.
        patch_pos = torch.nn.functional.interpolate(
            patch_pos,
            size=(target_grid_h, target_grid_w),
            mode="bicubic",
            align_corners=False,
        )

        # [1, C, H, W] -> [1, H*W, C]
        patch_pos = patch_pos.permute(
            0, 2, 3, 1
        ).reshape(
            1,
            target_grid_h * target_grid_w,
            hidden_dim,
        )

        resized_pos = torch.cat(
            [cls_pos, patch_pos],
            dim=1,
        ).to(
            device=original_device,
            dtype=original_dtype,
        )

    # Critical:
    # Re-register as TRAINABLE parameter.
    embeddings.position_embeddings = torch.nn.Parameter(
        resized_pos.contiguous(),
        requires_grad=True,
    )

    expected_tokens = 1 + target_grid_h * target_grid_w

    if embeddings.position_embeddings.shape[1] != expected_tokens:
        raise RuntimeError(
            "Unexpected resized positional embedding shape: "
            f"{tuple(embeddings.position_embeddings.shape)}"
        )

    print(
        "DINOv2 positional embeddings pre-resized: "
        f"{source_grid}x{source_grid} -> "
        f"{target_grid_h}x{target_grid_w}; "
        "position embeddings remain trainable.",
        flush=True,
    )


def _bicubic_kernel(distance: torch.Tensor) -> torch.Tensor:
    """PyTorch bicubic convolution kernel (a = -0.75)."""
    coefficient = -0.75
    distance = distance.abs()
    inner = (
        (coefficient + 2.0) * distance**3
        - (coefficient + 3.0) * distance**2
        + 1.0
    )
    outer = (
        coefficient * distance**3
        - 5.0 * coefficient * distance**2
        + 8.0 * coefficient * distance
        - 4.0 * coefficient
    )
    return torch.where(
        distance <= 1.0,
        inner,
        torch.where(distance < 2.0, outer, torch.zeros_like(distance)),
    )


def _build_bicubic_interpolation_matrix(
    source_size: int,
    target_size: int,
) -> torch.Tensor:
    """Build the align_corners=False 1-D bicubic resize matrix on CPU."""
    if source_size <= 0 or target_size <= 0:
        raise ValueError(
            f"Interpolation sizes must be positive, got {source_size} -> "
            f"{target_size}."
        )
    if source_size == target_size:
        return torch.eye(source_size, dtype=torch.float32)

    scale = source_size / target_size
    target_coordinates = torch.arange(target_size, dtype=torch.float32)
    source_coordinates = (target_coordinates + 0.5) * scale - 0.5
    source_floor = torch.floor(source_coordinates).to(torch.long)
    offsets = torch.arange(-1, 3, dtype=torch.long)
    source_indices = source_floor[:, None] + offsets[None, :]
    weights = _bicubic_kernel(
        source_coordinates[:, None] - source_indices.to(torch.float32)
    )

    # Bicubic uses border replication outside the source grid.
    source_indices.clamp_(0, source_size - 1)
    matrix = torch.zeros(target_size, source_size, dtype=torch.float32)
    rows = torch.arange(target_size)
    for column in range(4):
        matrix[rows, source_indices[:, column]] += weights[:, column]
    return matrix


def configure_full_deterministic_model_matrix_interpolation(model):
    """
    Keep the original DINOv2 position embedding trainable and replace its
    bicubic resize with separable deterministic matrix multiplications.

    Each interpolation matrix is built once per source/target size and device,
    then cached. The position embedding remains the original model parameter.
    """
    embeddings = model.dav2.backbone.embeddings
    position_embeddings = embeddings.position_embeddings
    num_source_patches = position_embeddings.shape[1] - 1
    source_grid = math.isqrt(num_source_patches)
    if source_grid * source_grid != num_source_patches:
        raise ValueError(
            "Expected a square pretrained positional grid, but got "
            f"{num_source_patches} patch positions."
        )

    matrix_cache = {}

    def interpolation_matrix(source_size, target_size, device):
        key = (source_size, target_size, device.type, device.index)
        matrix = matrix_cache.get(key)
        if matrix is None:
            matrix = _build_bicubic_interpolation_matrix(
                source_size,
                target_size,
            ).to(device=device)
            matrix_cache[key] = matrix
        return matrix

    def interpolate_pos_encoding(self, tokens, height, width):
        num_patches = tokens.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if (
            not torch.jit.is_tracing()
            and num_patches == num_positions
            and height == width
        ):
            return self.position_embeddings

        patch_size = self.patch_size
        if isinstance(patch_size, int):
            patch_height = patch_width = patch_size
        else:
            patch_height, patch_width = patch_size
        target_grid_height = height // patch_height
        target_grid_width = width // patch_width

        current_source_grid = math.isqrt(num_positions)
        if current_source_grid * current_source_grid != num_positions:
            raise ValueError(
                "Expected a square positional grid, but got "
                f"{num_positions} patch positions."
            )

        cls_pos = self.position_embeddings[:, :1]
        patch_pos = self.position_embeddings[:, 1:]
        hidden_dim = patch_pos.shape[-1]
        target_dtype = patch_pos.dtype
        patch_pos = patch_pos.reshape(
            1,
            current_source_grid,
            current_source_grid,
            hidden_dim,
        ).permute(0, 3, 1, 2)

        # DINOv2 interpolates position embeddings in float32. Disable autocast
        # explicitly so the matrix path preserves that behaviour.
        with torch.autocast(device_type=patch_pos.device.type, enabled=False):
            height_matrix = interpolation_matrix(
                current_source_grid,
                target_grid_height,
                patch_pos.device,
            )
            width_matrix = interpolation_matrix(
                current_source_grid,
                target_grid_width,
                patch_pos.device,
            )
            resized_pos = torch.matmul(height_matrix, patch_pos.float())
            resized_pos = torch.matmul(
                resized_pos,
                width_matrix.transpose(0, 1),
            )

        resized_pos = resized_pos.to(dtype=target_dtype)
        resized_pos = resized_pos.permute(0, 2, 3, 1).reshape(
            1,
            target_grid_height * target_grid_width,
            hidden_dim,
        )
        return torch.cat((cls_pos, resized_pos), dim=1)

    embeddings.interpolate_pos_encoding = MethodType(
        interpolate_pos_encoding,
        embeddings,
    )
    position_embeddings.requires_grad_(True)
    print(
        "DINOv2 positional interpolation replaced with cached bicubic "
        f"matrices; original {source_grid}x{source_grid} position embedding "
        "remains trainable.",
        flush=True,
    )


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
    model = model.to(
            device,
        )
    if args.full_deterministic:
        configure_full_deterministic_model_matrix_interpolation(model)
        print(
            "Full deterministic mode: strict algorithms and deterministic "
            "DINOv2 position interpolation; position embeddings trainable",
            flush=True,
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
