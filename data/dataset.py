# Stanford 2D-3D-S PriorBIM 数据集：负责数据加载、校验并调用 transform 完成原有增强。
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transform import Compose, RandomHorizontalFlip, RandomRGBGainBias


class S23PriorBIMDataset(Dataset):
    """
    Stanford 2D-3D-S Area_1 + BIMSyn prepared dataset.

    Expected layout
    ---------------
    dataset_root/
        metadata.json
        manifests/
            train.jsonl
            val.jsonl
            test.jsonl
        samples/
            office_1/
                camera_xxx_office_1_frame_0.npz
            ...

    Each NPZ contains
    -----------------
    sample_schema_version
    intrinsic
    da3_depth_raw
    da3_focal_scale
    bim_depth
    bim_valid
    gt_depth
    gt_valid

    Important
    ---------
    DA3 metric depth is reconstructed as:

        da3_depth =
            float32(da3_depth_raw)
            * da3_focal_scale

    This reproduces the previous PriorBIMDA convention.
    """

    def __init__(
        self,
        dataset_root: str | Path,
        s23_root: str | Path,
        split: str,
        *,
        augment: bool | None = None,
        color_jitter: float = 0.0,
        horizontal_flip_probability: float = 0.0,
    ):
        super().__init__()

        self.dataset_root = (
            Path(dataset_root)
            .expanduser()
            .resolve()
        )

        self.s23_root = (
            Path(s23_root)
            .expanduser()
            .resolve()
        )

        self.split = str(split)

        if self.split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(
                f"Unknown split: {self.split}"
            )

        # Default:
        # augment training only.
        if augment is None:
            augment = self.split == "train"

        self.augment = bool(augment)

        if (
            self.augment
            and self.split != "train"
        ):
            raise ValueError(
                "Augmentation is train-only"
            )

        self.color_jitter = float(
            color_jitter
        )

        self.horizontal_flip_probability = float(
            horizontal_flip_probability
        )

        if not (
            0.0
            <= self.horizontal_flip_probability
            <= 1.0
        ):
            raise ValueError(
                "horizontal_flip_probability "
                "must be in [0, 1]"
            )

        if self.color_jitter < 0:
            raise ValueError(
                "color_jitter must be >= 0"
            )

        transforms = []

        if self.augment and self.color_jitter > 0:
            transforms.append(
                RandomRGBGainBias(
                    gain_range=self.color_jitter,
                    bias_range=self.color_jitter,
                    p=1.0,
                )
            )

        if (
            self.augment
            and self.horizontal_flip_probability > 0
        ):
            transforms.append(
                RandomHorizontalFlip(
                    p=self.horizontal_flip_probability
                )
            )

        self.transform = Compose(transforms)

        target_shape = (504, 504)


        self.height, self.width = (
            target_shape
        )

        # ---------------------------------------------------------
        # Manifest
        # ---------------------------------------------------------

        manifest_path = (
            self.dataset_root
            / "manifests"
            / f"{self.split}.jsonl"
        )

        if not manifest_path.is_file():
            raise FileNotFoundError(
                manifest_path
            )

        self.records = [
            json.loads(line)
            for line
            in manifest_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not self.records:
            raise ValueError(
                f"Empty manifest: "
                f"{manifest_path}"
            )

        # ---------------------------------------------------------
        # Basic consistency checks
        # ---------------------------------------------------------

        ids = [
            str(record["id"])
            for record in self.records
        ]

        if len(ids) != len(set(ids)):
            raise ValueError(
                "Duplicate sample IDs "
                f"in {manifest_path}"
            )

        for record in self.records:
            if record.get("split") != self.split:
                raise ValueError(
                    f"{record['id']}: "
                    f"manifest split mismatch"
                )

    # =============================================================
    # Paths
    # =============================================================

    def _sample_path(
        self,
        record: dict,
    ) -> Path:
        path = Path(record["sample"])

        if not path.is_absolute():
            path = (
                self.dataset_root
                / path
            )

        return path

    def _rgb_path(
        self,
        record: dict,
    ) -> Path:
        path = Path(record["rgb"])

        if not path.is_absolute():
            path = (
                self.s23_root
                / path
            )

        return path

    # =============================================================
    # RGB
    # =============================================================

    def _load_rgb(
        self,
        path: Path,
    ) -> np.ndarray:

        image = cv2.imread(
            str(path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read RGB: {path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        if (
            image.shape[0] != self.height
            or image.shape[1] != self.width
        ):
            # PriorDA-style input:
            # use bicubic for RGB.
            image = cv2.resize(
                image,
                (
                    self.width,
                    self.height,
                ),
                interpolation=cv2.INTER_CUBIC,
            )

        rgb = (
            image.astype(np.float32)
            / 255.0
        )

        # HWC -> CHW
        return np.ascontiguousarray(
            rgb.transpose(2, 0, 1)
        )

    # =============================================================
    # NPZ
    # =============================================================

    def _load_sample(
        self,
        path: Path,
    ) -> dict[str, np.ndarray]:

        if not path.is_file():
            raise FileNotFoundError(
                path
            )

        with np.load(
            path,
            allow_pickle=False,
        ) as item:

            required = {
                "sample_schema_version",
                "intrinsic",
                "da3_depth_raw",
                "da3_focal_scale",
                "bim_depth",
                "bim_valid",
                "gt_depth",
                "gt_valid",
            }

            missing = (
                required
                - set(item.files)
            )

            if missing:
                raise ValueError(
                    f"{path}: missing keys "
                    f"{sorted(missing)}"
                )

            intrinsic = item[
                "intrinsic"
            ].astype(
                np.float32
            )

            da3_raw = item[
                "da3_depth_raw"
            ].astype(
                np.float32
            )

            focal_scale = float(
                item[
                    "da3_focal_scale"
                ].item()
            )

            bim_depth = item[
                "bim_depth"
            ].astype(
                np.float32
            )

            bim_valid = (
                item["bim_valid"]
                .astype(np.float32)
            )

            gt_depth = item[
                "gt_depth"
            ].astype(
                np.float32
            )

            gt_valid = (
                item["gt_valid"]
                .astype(np.float32)
            )

        # ---------------------------------------------------------
        # DA3 metric focal correction
        # ---------------------------------------------------------

        if (
            not np.isfinite(focal_scale)
            or focal_scale <= 0
        ):
            raise ValueError(
                f"{path}: invalid "
                "da3_focal_scale="
                f"{focal_scale}"
            )

        da3_depth = (
            da3_raw
            * focal_scale
        )

        # ---------------------------------------------------------
        # Contracts
        # ---------------------------------------------------------

        shape = (
            self.height,
            self.width,
        )

        for name, value in {
            "da3_depth": da3_depth,
            "bim_depth": bim_depth,
            "bim_valid": bim_valid,
            "gt_depth": gt_depth,
            "gt_valid": gt_valid,
        }.items():

            if value.shape != shape:
                raise ValueError(
                    f"{path}: "
                    f"{name}.shape="
                    f"{value.shape}, "
                    f"expected={shape}"
                )

        if intrinsic.shape != (3, 3):
            raise ValueError(
                f"{path}: intrinsic "
                f"shape={intrinsic.shape}"
            )

        if (
            not np.isfinite(
                da3_depth
            ).all()
            or np.any(
                da3_depth <= 0
            )
        ):
            raise ValueError(
                f"{path}: invalid DA3 depth"
            )

        bim_valid = (
            bim_valid > 0
        )

        gt_valid = (
            gt_valid > 0
        )

        if np.any(
            bim_valid
            & (
                ~np.isfinite(bim_depth)
                | (bim_depth <= 0)
            )
        ):
            raise ValueError(
                f"{path}: valid BIM "
                "pixels contain bad depth"
            )

        if np.any(
            (~bim_valid)
            & (bim_depth != 0)
        ):
            raise ValueError(
                f"{path}: invalid BIM "
                "pixels are not zero"
            )

        if np.any(
            gt_valid
            & (
                ~np.isfinite(gt_depth)
                | (gt_depth <= 0)
            )
        ):
            raise ValueError(
                f"{path}: valid GT "
                "pixels contain bad depth"
            )

        # Return [C,H,W].
        return {
            "intrinsic":
                intrinsic,

            "da3_depth":
                da3_depth[None],

            "bim_depth":
                bim_depth[None],

            "bim_valid":
                bim_valid.astype(
                    np.float32
                )[None],

            "gt_depth":
                gt_depth[None],

            "gt_valid":
                gt_valid.astype(
                    np.float32
                )[None],
        }

    # =============================================================
    # Augmentation
    # =============================================================

    def _augment(
        self,
        arrays: dict,
    ) -> dict:
        # 使用 Python random 保持原数据集的随机数来源、调用顺序和
        # DataLoader worker seeding 行为不变。
        return self.transform(arrays, rng=random)

    # =============================================================
    # Dataset API
    # =============================================================

    def __len__(self):
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ):
        record = self.records[index]

        sample_path = (
            self._sample_path(
                record
            )
        )

        rgb_path = (
            self._rgb_path(
                record
            )
        )

        arrays = self._load_sample(
            sample_path
        )

        arrays["rgb"] = (
            self._load_rgb(
                rgb_path
            )
        )

        if self.augment:
            arrays = self._augment(
                arrays
            )

        # ---------------------------------------------------------
        # numpy -> torch
        # ---------------------------------------------------------

        output = {
            key: torch.from_numpy(
                np.ascontiguousarray(
                    value
                )
            )
            for key, value
            in arrays.items()
        }

        # ---------------------------------------------------------
        # Metadata stays Python-native.
        # ---------------------------------------------------------

        output.update(
            {
                "sample_id":
                    str(record["id"]),

                "region":
                    str(record["region"]),

                "camera_uuid":
                    str(
                        record[
                            "camera_uuid"
                        ]
                    ),

                "frame_number":
                    int(
                        record[
                            "frame_number"
                        ]
                    ),
            }
        )

        return output


# ================================================================
# DataLoader helper
# ================================================================

def build_area1_dataloader(
    *,
    dataset_root,
    s23_root,
    split,
    batch_size,
    num_workers=8,
    shuffle=None,
    augment=None,
    color_jitter=0.0,
    horizontal_flip_probability=0.0,
    pin_memory=True,
    persistent_workers=True,
    drop_last=None,
):
    dataset = S23PriorBIMDataset(
        dataset_root=dataset_root,
        s23_root=s23_root,
        split=split,
        augment=augment,
        color_jitter=color_jitter,
        horizontal_flip_probability=(
            horizontal_flip_probability
        ),
    )

    if shuffle is None:
        shuffle = split == "train"

    if drop_last is None:
        drop_last = split == "train"

    if num_workers == 0:
        persistent_workers = False

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(
            persistent_workers
        ),
        drop_last=drop_last,
    )

    return loader
