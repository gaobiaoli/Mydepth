# 数据增强基础接口：定义多模态样本、空间字段、概率校验和组合式 transform。
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TypeAlias

import numpy as np

Sample: TypeAlias = dict[str, np.ndarray]


# 所有与 image plane 对齐的字段。
SPATIAL_KEYS = (
    "rgb",
    "da3_depth",
    "bim_depth",
    "bim_valid",
    "gt_depth",
    "gt_valid",
)


def validate_probability(
    p: float,
    *,
    name: str = "p",
) -> float:
    p = float(p)

    if not 0.0 <= p <= 1.0:
        raise ValueError(
            f"{name} must be in [0, 1], got {p}"
        )

    return p


class Transform(ABC):
    """
    Base class for sample-level augmentation.

    Every transform receives the complete multimodal sample and
    an explicit NumPy RNG.

    A transform may modify the sample in-place and return it.
    """

    @abstractmethod
    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:
        raise NotImplementedError


class Identity(Transform):

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:
        return sample


class Compose(Transform):
    """
    Sequentially apply sample-level transforms.
    """

    def __init__(
        self,
        transforms: Sequence[Transform],
    ):
        self.transforms = tuple(transforms)

    def __call__(
        self,
        sample: Sample,
        *,
        rng: np.random.Generator,
    ) -> Sample:

        for transform in self.transforms:
            sample = transform(
                sample,
                rng=rng,
            )

        return sample

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        body = ",\n".join(
            f"    {transform!r}"
            for transform in self.transforms
        )

        return (
            f"{self.__class__.__name__}([\n"
            f"{body}\n"
            f"])"
        )
