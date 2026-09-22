from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import (
    Sample,
    Transform,
    validate_probability,
)


def _get_prediction(
    sample: Sample,
    key: str,
) -> np.ndarray:

    if key not in sample:
        raise KeyError(f"Sample does not contain '{key}'")

    depth = sample[key]

    if depth.ndim != 3 or depth.shape[0] != 1:
        raise ValueError("Expected prediction shape [1,H,W], " f"got {depth.shape}")

    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError(f"{key} contains invalid depth")

    return depth


@dataclass
class RandomPredictionScale(Transform):
    """
    Global multiplicative perturbation of
    the raw depth prediction.

        D_pred' = D_pred * exp(epsilon)

        epsilon ~ U(
            -max_log_scale,
            +max_log_scale
        )

    This simulates frame-level metric-scale
    error in the base MDE prediction.
    """

    max_log_scale: float = 0.2
    p: float = 1.0
    key: str = "da3_depth"

    def __post_init__(self):
        self.max_log_scale = float(self.max_log_scale)
        self.p = validate_probability(self.p)
        self.key = str(self.key)

        if self.max_log_scale < 0:
            raise ValueError("max_log_scale must be >= 0")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if self.p < 1.0 and rng.random() >= self.p:
            return sample

        depth = _get_prediction(
            sample,
            self.key,
        )

        epsilon = rng.uniform(
            -self.max_log_scale,
            self.max_log_scale,
        )

        scale = np.float32(np.exp(epsilon))

        sample[self.key] = (
            depth.astype(
                np.float32,
                copy=False,
            )
            * scale
        ).astype(
            np.float32,
            copy=False,
        )

        return sample
