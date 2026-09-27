"""Frozen scale-shift calibration followed by weighted dense refinement."""
from __future__ import annotations

from pathlib import Path

import torch
from torch.nn import functional as F

from loss import depth_weights, masked_frame_mean
from model.dense2dense import PriorBIMDA as DenseRefiner
from model.mymodel1_onlyscale_shift import PriorBIMDA as ScaleShiftModel
from model.scale_shift_dense2dense_noweight import PriorBIMDA as CompositeModel


class PriorBIMDA(CompositeModel):
    """Scale-shift + dense-to-dense using the original weighted depth loss."""

    @classmethod
    def from_pretrained(
        cls,
        local_files_only=False,
        calibration_checkpoint: str | Path | None = None,
    ):
        calibrator = ScaleShiftModel.from_pretrained(
            local_files_only=local_files_only
        )
        if calibration_checkpoint is not None:
            calibrator.load_checkpoint(calibration_checkpoint)
        refiner = DenseRefiner.from_pretrained(local_files_only=local_files_only)
        return cls(calibrator, refiner)

    def compute_loss(self, output, batch):
        prediction = output["depth"].float()
        target = batch["gt_depth"].float()
        base_depth = output["calibrated_depth"].detach().float()
        valid = (
            (batch["gt_valid"] > 0)
            & (target > 0)
            & (prediction > 0)
            & (base_depth > 0)
            & torch.isfinite(target)
            & torch.isfinite(prediction)
            & torch.isfinite(base_depth)
        )
        weights = depth_weights(batch) * valid.float()
        log_error = (
            prediction.clamp_min(1e-6).log()
            - target.clamp_min(1e-6).log()
        ).abs()
        pixel_loss = (log_error * weights).sum() / weights.sum().clamp_min(1)
        frame_loss = masked_frame_mean(log_error, weights)
        depth_loss = 0.5 * (pixel_loss + frame_loss)

        log_target = (
            target.clamp_min(1e-6).log()
            - base_depth.clamp_min(1e-6).log()
        )
        dense_loss = masked_frame_mean(
            F.smooth_l1_loss(
                output["dense_log_residual"].float(),
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
