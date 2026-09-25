"""Dense-to-dense model with uniform valid-pixel depth loss."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from loss import masked_frame_mean
from model.dense2dense import PriorBIMDA as Dense2Dense


class PriorBIMDA(Dense2Dense):
    """Keep the dense-to-dense network and remove only depth pixel weights."""

    def compute_loss(self, output, batch):
        prediction = output["depth"].float()
        target = batch["gt_depth"].float()
        valid = (
            (batch["gt_valid"] > 0)
            & (target > 0)
            & (prediction > 0)
            & torch.isfinite(target)
            & torch.isfinite(prediction)
        )
        weights = valid.float()
        log_error = (
            prediction.clamp_min(1e-6).log()
            - target.clamp_min(1e-6).log()
        ).abs()
        pixel_loss = (log_error * weights).sum() / weights.sum().clamp_min(1)
        frame_loss = masked_frame_mean(log_error, weights)
        depth_loss = 0.5 * (pixel_loss + frame_loss)

        log_target = (
            target.clamp_min(1e-6).log()
            - batch["da3_depth"].detach().float().clamp_min(1e-6).log()
        )
        dense_loss = masked_frame_mean(
            F.smooth_l1_loss(
                output["log_scale"].float(),
                log_target,
                reduction="none",
                beta=0.02,
            ),
            valid,
        )
        return {
            "total": depth_loss + 0.5 * dense_loss,
            "depth": depth_loss,
            "dense": dense_loss,
        }
