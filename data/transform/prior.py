from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import (
    Sample,
    Transform,
    randint_inclusive,
    validate_probability,
)


def _get_bim(
    sample: Sample,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    if "bim_depth" not in sample:
        raise KeyError("Sample does not contain " "'bim_depth'")

    if "bim_valid" not in sample:
        raise KeyError("Sample does not contain " "'bim_valid'")

    depth = sample["bim_depth"]

    valid = sample["bim_valid"] > 0.5

    if depth.shape != valid.shape:
        raise ValueError("bim_depth and bim_valid " "must have the same shape")

    return depth, valid


def _enforce_bim_contract(
    sample: Sample,
) -> Sample:

    depth, valid = _get_bim(sample)

    depth = depth.astype(
        np.float32,
        copy=False,
    )

    depth = np.where(
        valid,
        depth,
        0.0,
    ).astype(
        np.float32,
        copy=False,
    )

    sample["bim_depth"] = depth
    sample["bim_valid"] = valid.astype(np.float32)

    return sample


@dataclass
class RandomBIMScale(Transform):
    """
    Global multiplicative BIM depth perturbation.

        D_bim' = D_bim * exp(epsilon)

    epsilon ~ U(-max_log_scale,
                 +max_log_scale)

    Mainly useful as robustness augmentation /
    stress test rather than a literal BIM error
    model.
    """

    max_log_scale: float = 0.03
    p: float = 0.3

    def __post_init__(self):
        self.max_log_scale = float(self.max_log_scale)
        self.p = validate_probability(self.p)

        if self.max_log_scale < 0:
            raise ValueError("max_log_scale must be >= 0")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        depth, valid = _get_bim(sample)

        scale = np.exp(
            rng.uniform(
                -self.max_log_scale,
                self.max_log_scale,
            )
        )

        depth = depth.copy()

        depth[valid] *= np.float32(scale)

        sample["bim_depth"] = depth

        return _enforce_bim_contract(sample)


@dataclass
class RandomBIMLogNoise(Transform):
    """
    Pixel-level multiplicative BIM noise.

        D' = D * exp(epsilon)
        epsilon ~ N(0, sigma^2)

    Positive depth is guaranteed.
    """

    sigma: float = 0.01
    p: float = 0.3

    def __post_init__(self):
        self.sigma = float(self.sigma)
        self.p = validate_probability(self.p)

        if self.sigma < 0:
            raise ValueError("sigma must be >= 0")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        depth, valid = _get_bim(sample)

        depth = depth.copy()

        # PriorBIMDA uses Python random for the probability decision and the
        # NumPy global RNG for the actual per-pixel noise.
        noise_rng = rng if hasattr(rng, "normal") else np.random
        noise = noise_rng.normal(
            0.0,
            self.sigma,
            size=depth.shape,
        ).astype(np.float32)

        depth[valid] *= np.exp(noise[valid])

        sample["bim_depth"] = depth

        return _enforce_bim_contract(sample)


@dataclass
class RandomBIMDropout(Transform):
    """
    Randomly invalidate individual BIM pixels.

    This is a generic robustness augmentation.
    Spatially structured holes are usually more
    realistic and should often be preferred.
    """

    dropout_probability: float = 0.05
    p: float = 0.3

    def __post_init__(self):
        self.dropout_probability = validate_probability(
            self.dropout_probability,
            name="dropout_probability",
        )

        self.p = validate_probability(self.p)

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        depth, valid = _get_bim(sample)

        drop = rng.random(valid.shape) < self.dropout_probability

        new_valid = valid & ~drop

        depth = depth.copy()
        depth[~new_valid] = 0.0

        sample["bim_depth"] = depth
        sample["bim_valid"] = new_valid.astype(np.float32)

        return sample


@dataclass
class RandomBIMRectHole(Transform):
    """
    Remove one or more rectangular regions
    from BIM prior.

    This represents spatially structured missing
    prior and is generally more meaningful than
    independent pixel dropout.
    """

    min_fraction: float = 0.05
    max_fraction: float = 0.20
    holes: int = 1
    p: float = 0.3

    def __post_init__(self):
        self.min_fraction = float(self.min_fraction)

        self.max_fraction = float(self.max_fraction)

        self.holes = int(self.holes)

        self.p = validate_probability(self.p)

        if not (0.0 < self.min_fraction <= self.max_fraction <= 1.0):
            raise ValueError("Require " "0 < min_fraction <= " "max_fraction <= 1")

        if self.holes < 1:
            raise ValueError("holes must be >= 1")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        depth, valid = _get_bim(sample)

        if depth.ndim != 3:
            raise ValueError("Expected BIM shape [1,H,W]")

        H = depth.shape[-2]
        W = depth.shape[-1]

        new_valid = valid.copy()

        for _ in range(self.holes):
            frac_h = rng.uniform(
                self.min_fraction,
                self.max_fraction,
            )

            frac_w = rng.uniform(
                self.min_fraction,
                self.max_fraction,
            )

            hole_h = max(
                1,
                round(H * frac_h),
            )

            hole_w = max(
                1,
                round(W * frac_w),
            )

            y0 = int(
                rng.integers(
                    0,
                    H - hole_h + 1,
                )
            )

            x0 = int(
                rng.integers(
                    0,
                    W - hole_w + 1,
                )
            )

            new_valid[
                ...,
                y0 : y0 + hole_h,
                x0 : x0 + hole_w,
            ] = False

        depth = depth.copy()
        depth[~new_valid] = 0.0

        sample["bim_depth"] = depth
        sample["bim_valid"] = new_valid.astype(np.float32)

        return sample


@dataclass
class RandomBIMSquareDropout(Transform):
    """PriorBIMDA fixed-area square BIM dropout."""

    fraction: float = 0.12
    p: float = 0.15

    def __post_init__(self):
        self.fraction = float(self.fraction)
        self.p = validate_probability(self.p)

        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must be in [0, 1]")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        depth, _ = _get_bim(sample)
        height, width = depth.shape[-2:]
        area = int(self.fraction * height * width)
        side = max(1, int(np.sqrt(area)))
        y = randint_inclusive(rng, 0, max(0, height - side))
        x = randint_inclusive(rng, 0, max(0, width - side))

        sample["bim_valid"][..., y : y + side, x : x + side] = 0
        sample["bim_depth"][..., y : y + side, x : x + side] = 0

        if "bim_normals" in sample:
            sample["bim_normals"][..., y : y + side, x : x + side] = 0
        if "bim_edge" in sample:
            sample["bim_edge"][..., y : y + side, x : x + side] = 1

        return _enforce_bim_contract(sample)


@dataclass
class RandomBIMFullDropout(Transform):
    """Invalidate the complete BIM prior, matching PriorBIMDA."""

    p: float = 0.03

    def __post_init__(self):
        self.p = validate_probability(self.p)

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        _get_bim(sample)
        sample["bim_valid"][...] = 0
        sample["bim_depth"][...] = 0

        if "bim_normals" in sample:
            sample["bim_normals"][...] = 0
        if "bim_edge" in sample:
            sample["bim_edge"][...] = 1

        return _enforce_bim_contract(sample)


@dataclass
class RandomBIMEdgeDilation(Transform):
    """Dilate the optional BIM edge map using PriorBIMDA's kernel."""

    pixels: int = 3
    p: float = 0.15

    def __post_init__(self):
        self.pixels = int(self.pixels)
        self.p = validate_probability(self.p)

        if self.pixels < 0:
            raise ValueError("pixels must be >= 0")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        # Consume the probability draw even when the compact MyDepth sample has
        # no bim_edge field, preserving all later PriorBIMDA RNG decisions.
        if rng.random() >= self.p or "bim_edge" not in sample:
            return sample

        kernel_size = max(1, self.pixels)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        sample["bim_edge"][0] = cv2.dilate(
            sample["bim_edge"][0].astype(np.float32),
            kernel,
        )
        return sample


def _translate_chw(
    value: np.ndarray,
    *,
    dx: int,
    dy: int,
    fill_value: float,
) -> np.ndarray:

    if value.ndim != 3:
        raise ValueError("Expected [C,H,W]")

    _, H, W = value.shape

    output = np.full_like(
        value,
        fill_value,
    )

    src_x0 = max(
        0,
        -dx,
    )
    src_x1 = min(
        W,
        W - dx,
    )

    dst_x0 = max(
        0,
        dx,
    )
    dst_x1 = min(
        W,
        W + dx,
    )

    src_y0 = max(
        0,
        -dy,
    )
    src_y1 = min(
        H,
        H - dy,
    )

    dst_y0 = max(
        0,
        dy,
    )
    dst_y1 = min(
        H,
        H + dy,
    )

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        return output

    output[
        ...,
        dst_y0:dst_y1,
        dst_x0:dst_x1,
    ] = value[
        ...,
        src_y0:src_y1,
        src_x0:src_x1,
    ]

    return output


@dataclass
class RandomBIMShift(Transform):
    """
    Simulate BIM-image registration error.

    Only BIM prior is translated.

    Camera intrinsic is intentionally NOT changed,
    because this models prior misregistration
    rather than camera motion.
    """

    max_dx: int = 5
    max_dy: int = 5
    p: float = 0.3

    def __post_init__(self):
        self.max_dx = int(self.max_dx)
        self.max_dy = int(self.max_dy)

        self.p = validate_probability(self.p)

        if self.max_dx < 0 or self.max_dy < 0:
            raise ValueError("max_dx/max_dy must be >= 0")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        if rng.random() >= self.p:
            return sample

        depth, valid = _get_bim(sample)

        dx = randint_inclusive(rng, -self.max_dx, self.max_dx)
        dy = randint_inclusive(rng, -self.max_dy, self.max_dy)

        depth = _translate_chw(
            depth,
            dx=dx,
            dy=dy,
            fill_value=0.0,
        )

        valid_float = valid.astype(np.float32)

        valid_float = _translate_chw(
            valid_float,
            dx=dx,
            dy=dy,
            fill_value=0.0,
        )

        sample["bim_depth"] = depth.astype(
            np.float32,
            copy=False,
        )

        sample["bim_valid"] = (valid_float > 0.5).astype(np.float32)

        for key in ("bim_normals", "bim_edge"):
            if key in sample:
                sample[key] = _translate_chw(
                    sample[key],
                    dx=dx,
                    dy=dy,
                    fill_value=0.0,
                )

        return _enforce_bim_contract(sample)
