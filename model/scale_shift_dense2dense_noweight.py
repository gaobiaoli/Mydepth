"""Frozen global scale-shift calibration followed by dense log refinement."""
from __future__ import annotations

import warnings
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from loss import masked_frame_mean
from model.dense2dense_noweight import PriorBIMDA as DenseRefiner
from model.mymodel1_onlyscale_shift import PriorBIMDA as ScaleShiftModel


class PriorBIMDA(nn.Module):
    """Apply a frozen scale-shift model, then train a dense residual refiner."""

    def __init__(self, calibrator, refiner):
        super().__init__()
        self.calibrator = calibrator
        self.refiner = refiner
        self._warned_invalid_calibration = False
        self._freeze_calibrator()

    @property
    def dav2(self):
        """Expose the trainable DAv2 for the shared deterministic setup helper."""
        return self.refiner.dav2

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

    def _freeze_calibrator(self):
        self.calibrator.requires_grad_(False)
        self.calibrator.eval()

    def train(self, mode=True):
        super().train(mode)
        self.calibrator.eval()
        return self

    def forward(self, rgb, da3_depth, bim_depth, bim_valid):
        with torch.no_grad():
            calibration = self.calibrator(
                rgb,
                da3_depth,
                bim_depth,
                bim_valid,
            )
        log_scale = calibration["log_scale"].detach().float()
        relative_shift = calibration["relative_shift"].detach().float()
        invalid = ~torch.isfinite(log_scale) | ~torch.isfinite(relative_shift)
        log_scale = torch.nan_to_num(log_scale, nan=0.0, posinf=0.0, neginf=0.0)
        relative_shift = torch.nan_to_num(
            relative_shift,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        raw_depth = da3_depth.detach().float()
        reference_depth = calibration["reference_depth"].detach().float()
        metric_shift = reference_depth * relative_shift
        calibration_scaled_depth = raw_depth * log_scale.exp()
        minimum_depth = 1e-3 + torch.finfo(raw_depth.dtype).eps
        calibrated_depth = (calibration_scaled_depth + metric_shift).clamp(
            minimum_depth, 128
        )

        # A finite but extreme correction can still overflow during exp or
        # addition. In that rare case use the valid raw DA3 depth for the
        # affected frame instead of feeding zero/NaN depth to the refiner.
        valid_frame = torch.isfinite(calibrated_depth).flatten(1).all(1)
        if (
            bool(invalid.any()) or not bool(valid_frame.all())
        ) and not self._warned_invalid_calibration:
            warnings.warn(
                "Frozen scale-shift model produced a non-finite correction; "
                "using the identity correction for the affected value/frame.",
                RuntimeWarning,
            )
            self._warned_invalid_calibration = True
        if not bool(valid_frame.all()):
            fallback = raw_depth.clamp(minimum_depth, 128)
            calibrated_depth = torch.where(
                valid_frame.view(-1, 1, 1, 1),
                calibrated_depth,
                fallback,
            )
        refined = self.refiner(
            rgb,
            calibrated_depth,
            bim_depth,
            bim_valid,
        )
        return {
            "depth": refined["depth"],
            "scaled_depth": calibrated_depth,
            "calibrated_depth": calibrated_depth,
            "calibration_scaled_depth": calibration_scaled_depth,
            "calibration_log_scale": log_scale,
            "calibration_shift": metric_shift,
            "dense_log_residual": refined["log_scale"],
            "log_scale": refined["log_scale"],
        }

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

    def parameter_groups(self, factor=1.0):
        return self.refiner.parameter_groups(factor)

    def load_checkpoint(self, path: str | Path):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state = (
            checkpoint.get("model", checkpoint)
            if isinstance(checkpoint, Mapping)
            else checkpoint
        )
        self.load_state_dict(state, strict=True)
        self._freeze_calibrator()
