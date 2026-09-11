# 公共损失模块：集中提供 PriorBIMDA 各独立模型共享的尺度、深度与 residual 损失实现。
from __future__ import annotations

import torch
from torch.nn import functional as F


@torch.no_grad()
def absrel_optimal_log_scale(da3_depth, gt_depth, gt_valid, min_pixels=100):
    """Per-image scale that minimizes AbsRel; used only as a training target."""
    valid = (
        (gt_valid > 0)
        & (da3_depth > 0)
        & (gt_depth > 0)
        & torch.isfinite(da3_depth)
        & torch.isfinite(gt_depth)
    )
    supported = valid.flatten(1).sum(1) >= min_pixels
    ratios = torch.where(
        valid,
        gt_depth / da3_depth.clamp_min(1e-8),
        torch.full_like(gt_depth, torch.inf),
    ).flatten(1)
    weights = torch.where(
        valid,
        da3_depth / gt_depth.clamp_min(1e-8),
        torch.zeros_like(gt_depth),
    ).flatten(1)

    ratios, order = ratios.sort(dim=1)
    weights = weights.gather(1, order)
    index = (weights.cumsum(1) >= 0.5 * weights.sum(1, keepdim=True)).to(torch.int8).argmax(1)
    scale = ratios.gather(1, index[:, None]).squeeze(1)
    supported &= torch.isfinite(scale) & (scale > 0)
    log_scale = torch.where(supported, scale.clamp_min(1e-8).log(), torch.zeros_like(scale))
    return log_scale.view(-1, 1, 1, 1), supported


def depth_weights(batch):
    """Weights used by the best run. Furniture weighting is optional in this release."""
    gt = batch["gt_depth"]
    valid = batch["gt_valid"] > 0
    weight = 1 + 2 * (gt < 1).float()

    if "furniture_mask" in batch:
        weight = weight + (batch["furniture_mask"] > 0).float()

    tolerance = torch.maximum(torch.full_like(batch["bim_depth"], 0.10),
                              0.05 * batch["bim_depth"])
    conflict = (
        valid
        & (batch["bim_valid"] > 0)
        & (gt > 0)
        & (batch["bim_depth"] > 0)
        & (gt < batch["bim_depth"] - tolerance)
    )
    return valid.float() * (weight + conflict.float())


def masked_downsample(value, valid, size):
    valid = valid.float()
    support = F.adaptive_avg_pool2d(valid, size)
    value = F.adaptive_avg_pool2d(value * valid, size) / support.clamp_min(1e-8)
    return torch.where(support > 0, value, torch.zeros_like(value)), support > 0


def masked_frame_mean(value, valid):
    valid = valid.float()
    numerator = (value * valid).flatten(1).sum(1)
    denominator = valid.flatten(1).sum(1).clamp_min(1)
    return (numerator / denominator).mean()


def priorbim_loss(output, batch, equivariance_error=None):
    """The five active losses of the adapter + 3 ResBlocks experiment."""
    prediction = output["depth"].float()
    target = batch["gt_depth"].float()
    valid = (
        (batch["gt_valid"] > 0)
        & (target > 0)
        & (prediction > 0)
        & torch.isfinite(target)
        & torch.isfinite(prediction)
    )
    weights = depth_weights(batch) * valid.float()
    log_error = (prediction.clamp_min(1e-6).log() - target.clamp_min(1e-6).log()).abs()

    pixel_loss = (log_error * weights).sum() / weights.sum().clamp_min(1)
    frame_numerator = (log_error * weights).flatten(1).sum(1)
    frame_denominator = weights.flatten(1).sum(1)
    available = frame_denominator > 0
    frame_loss = (frame_numerator[available] / frame_denominator[available]).mean()
    depth_loss = 0.5 * (pixel_loss + frame_loss)

    oracle_scale, supported = absrel_optimal_log_scale(
        batch["da3_depth"].float(), target, batch["gt_valid"]
    )
    scale_error = F.smooth_l1_loss(
        output["log_scale"].flatten(1).mean(1),
        oracle_scale.flatten(1).mean(1),
        reduction="none",
        beta=0.02,
    )
    scale_loss = scale_error[supported].mean() if bool(supported.any()) else prediction.sum() * 0

    oracle_depth = batch["da3_depth"].float() * oracle_scale.exp()
    residual_target = target.clamp_min(1e-6).log() - oracle_depth.clamp_min(1e-6).log()
    target_mean = (residual_target * valid).flatten(1).sum(1) / valid.flatten(1).sum(1).clamp_min(1)
    residual_target = residual_target - target_mean[:, None, None, None]

    native = output["log_residual_native"].float()
    native_target, native_valid = masked_downsample(residual_target, valid, native.shape[-2:])
    residual_loss = masked_frame_mean(
        F.smooth_l1_loss(native, native_target, reduction="none", beta=0.02),
        native_valid,
    )
    zero_mean_loss = native.mean(dim=(1, 2, 3)).abs().mean()
    equivariance_loss = (
        equivariance_error.float().square().mean()
        if equivariance_error is not None
        else prediction.sum() * 0
    )

    total = (
        depth_loss
        + 0.5 * scale_loss
        + 0.5 * residual_loss
        + 0.1 * zero_mean_loss
        + 0.1 * equivariance_loss
    )
    return {
        "total": total,
        "depth": depth_loss,
        "scale": scale_loss,
        "residual": residual_loss,
        "zero_mean": zero_mean_loss,
        "equivariance": equivariance_loss,
    }
