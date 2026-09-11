from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .base import (
    SPATIAL_KEYS,
    Sample,
    Transform,
    validate_probability,
)


def _get_hw(
    sample: Sample,
) -> tuple[int, int]:

    if "rgb" not in sample:
        raise KeyError("Sample must contain rgb")

    rgb = sample["rgb"]

    if rgb.ndim != 3:
        raise ValueError(f"Invalid rgb shape: " f"{rgb.shape}")

    return (
        int(rgb.shape[-2]),
        int(rgb.shape[-1]),
    )


@dataclass
class RandomHorizontalFlip(Transform):
    p: float = 0.5

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

        height, width = _get_hw(sample)

        for key in SPATIAL_KEYS:
            if key not in sample:
                continue

            value = sample[key]

            if value.shape[-2:] != (height, width):
                raise ValueError(
                    f"{key} spatial shape "
                    f"{value.shape[-2:]} "
                    f"!= {(height, width)}"
                )

            sample[key] = value[..., ::-1].copy()

        if "intrinsic" in sample:
            K = sample["intrinsic"].copy()

            # x' = W - 1 - x
            K[0, 2] = width - 1 - K[0, 2]

            sample["intrinsic"] = K

        return sample


@dataclass
class RandomCrop(Transform):
    """
    Random fixed-size crop.

    All image-aligned fields are cropped
    synchronously.

    Principal point is translated accordingly.
    """

    height: int
    width: int

    def __post_init__(self):
        self.height = int(self.height)
        self.width = int(self.width)

        if self.height <= 0 or self.width <= 0:
            raise ValueError("Crop size must be positive")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        H, W = _get_hw(sample)

        if self.height > H or self.width > W:
            raise ValueError(
                "Crop size "
                f"{(self.height, self.width)} "
                "is larger than image "
                f"{(H, W)}"
            )

        y0 = int(
            rng.integers(
                0,
                H - self.height + 1,
            )
        )

        x0 = int(
            rng.integers(
                0,
                W - self.width + 1,
            )
        )

        y1 = y0 + self.height
        x1 = x0 + self.width

        for key in SPATIAL_KEYS:
            if key not in sample:
                continue

            value = sample[key]

            if value.shape[-2:] != (H, W):
                raise ValueError(f"{key} spatial shape " "does not match RGB")

            sample[key] = value[
                ...,
                y0:y1,
                x0:x1,
            ].copy()

        if "intrinsic" in sample:
            K = sample["intrinsic"].copy()

            K[0, 2] -= x0
            K[1, 2] -= y0

            sample["intrinsic"] = K

        return sample


def _resize_chw(
    array: np.ndarray,
    width: int,
    height: int,
    interpolation: int,
) -> np.ndarray:

    if array.ndim != 3:
        raise ValueError("Expected [C,H,W], " f"got {array.shape}")

    hwc = array.transpose(
        1,
        2,
        0,
    )

    resized = cv2.resize(
        hwc,
        (width, height),
        interpolation=interpolation,
    )

    if resized.ndim == 2:
        resized = resized[
            ...,
            None,
        ]

    return np.ascontiguousarray(
        resized.transpose(
            2,
            0,
            1,
        )
    )


@dataclass
class Resize(Transform):
    """
    Deterministic multimodal resize.

    RGB:
        cubic interpolation

    DA3:
        linear interpolation

    BIM / GT depth:
        nearest interpolation, avoiding
        interpolation across invalid boundaries

    masks:
        nearest interpolation
    """

    height: int
    width: int

    def __post_init__(self):
        self.height = int(self.height)
        self.width = int(self.width)

        if self.height <= 0 or self.width <= 0:
            raise ValueError("Resize shape must be positive")

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        old_h, old_w = _get_hw(sample)

        if old_h == self.height and old_w == self.width:
            return sample

        if "rgb" in sample:
            sample["rgb"] = _resize_chw(
                sample["rgb"],
                self.width,
                self.height,
                cv2.INTER_CUBIC,
            ).astype(
                np.float32,
                copy=False,
            )

        if "da3_depth" in sample:
            sample["da3_depth"] = _resize_chw(
                sample["da3_depth"],
                self.width,
                self.height,
                cv2.INTER_LINEAR,
            ).astype(
                np.float32,
                copy=False,
            )

        for key in (
            "bim_depth",
            "gt_depth",
        ):
            if key in sample:
                sample[key] = _resize_chw(
                    sample[key],
                    self.width,
                    self.height,
                    cv2.INTER_NEAREST,
                ).astype(
                    np.float32,
                    copy=False,
                )

        for key in (
            "bim_valid",
            "gt_valid",
        ):
            if key in sample:
                mask = _resize_chw(
                    sample[key],
                    self.width,
                    self.height,
                    cv2.INTER_NEAREST,
                )

                sample[key] = (mask > 0.5).astype(np.float32)

        if "intrinsic" in sample:
            sx = self.width / old_w

            sy = self.height / old_h

            K = sample["intrinsic"].copy()

            K[0, 0] *= sx
            K[1, 1] *= sy
            K[0, 2] *= sx
            K[1, 2] *= sy

            sample["intrinsic"] = K

        return sample
