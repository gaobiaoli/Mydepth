from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import (
    Sample,
    Transform,
    validate_probability,
)


def _validate_rgb(
    sample: Sample,
) -> np.ndarray:

    if "rgb" not in sample:
        raise KeyError(
            "Sample does not contain 'rgb'"
        )

    rgb = sample["rgb"]

    if (
        rgb.ndim != 3
        or rgb.shape[0] != 3
    ):
        raise ValueError(
            "Expected rgb shape [3,H,W], "
            f"got {rgb.shape}"
        )

    return rgb


@dataclass
class RandomRGBGainBias(Transform):
    """
    RGB-only intensity perturbation.

    rgb' = rgb * gain + bias

    gain ~ U(1-gain_range, 1+gain_range)
    bias ~ U(-bias_range, +bias_range)
    """

    gain_range: float = 0.0
    bias_range: float = 0.0
    p: float = 1.0

    def __post_init__(self):
        self.gain_range = float(
            self.gain_range
        )
        self.bias_range = float(
            self.bias_range
        )
        self.p = validate_probability(
            self.p
        )

        if self.gain_range < 0:
            raise ValueError(
                "gain_range must be >= 0"
            )

        if self.bias_range < 0:
            raise ValueError(
                "bias_range must be >= 0"
            )

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        # p=1 时不额外消耗随机数，使 dataset 的 gain/bias/flip 抽样
        # 顺序与迁移到 transform 模块前完全一致。
        if self.p < 1.0 and rng.random() >= self.p:
            return sample

        rgb = _validate_rgb(
            sample
        ).astype(
            np.float32,
            copy=False,
        )

        gain = rng.uniform(
            1.0 - self.gain_range,
            1.0 + self.gain_range,
        )

        bias = rng.uniform(
            -self.bias_range,
            self.bias_range,
        )

        rgb = (
            rgb * np.float32(gain)
            + np.float32(bias)
        )

        sample["rgb"] = np.clip(
            rgb,
            0.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        return sample


@dataclass
class RandomGamma(Transform):
    """
    RGB-only gamma augmentation.

        rgb' = rgb ** gamma
    """

    gamma_min: float = 0.8
    gamma_max: float = 1.2
    p: float = 0.5

    def __post_init__(self):
        self.gamma_min = float(
            self.gamma_min
        )
        self.gamma_max = float(
            self.gamma_max
        )
        self.p = validate_probability(
            self.p
        )

        if self.gamma_min <= 0:
            raise ValueError(
                "gamma_min must be > 0"
            )

        if (
            self.gamma_max
            < self.gamma_min
        ):
            raise ValueError(
                "gamma_max must be >= "
                "gamma_min"
            )

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        rgb = _validate_rgb(
            sample
        ).astype(
            np.float32,
            copy=False,
        )

        gamma = rng.uniform(
            self.gamma_min,
            self.gamma_max,
        )

        sample["rgb"] = np.power(
            np.clip(
                rgb,
                0.0,
                1.0,
            ),
            np.float32(gamma),
        ).astype(
            np.float32,
            copy=False,
        )

        return sample


@dataclass
class RandomRGBGaussianNoise(
    Transform
):
    """
    Additive Gaussian sensor-like RGB noise.

        rgb' = rgb + N(0, sigma^2)
    """

    sigma: float = 0.01
    p: float = 0.3

    def __post_init__(self):
        self.sigma = float(
            self.sigma
        )
        self.p = validate_probability(
            self.p
        )

        if self.sigma < 0:
            raise ValueError(
                "sigma must be >= 0"
            )

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        rgb = _validate_rgb(
            sample
        ).astype(
            np.float32,
            copy=False,
        )

        noise = rng.normal(
            loc=0.0,
            scale=self.sigma,
            size=rgb.shape,
        ).astype(
            np.float32
        )

        sample["rgb"] = np.clip(
            rgb + noise,
            0.0,
            1.0,
        ).astype(
            np.float32,
            copy=False,
        )

        return sample
