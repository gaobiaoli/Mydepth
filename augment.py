from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np


Sample = dict[str, np.ndarray]


class Transform(ABC):
    @abstractmethod
    def __call__(
        self,
        sample: Sample,
        rng: np.random.Generator,
    ) -> Sample:
        pass


class Compose:
    def __init__(
        self,
        transforms: Sequence[Transform],
    ):
        self.transforms = list(transforms)

    def __call__(
        self,
        sample: Sample,
        rng: np.random.Generator,
    ) -> Sample:

        for transform in self.transforms:
            sample = transform(
                sample,
                rng=rng,
            )

        return sample