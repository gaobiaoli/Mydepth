# Stanford 2D-3D-S PriorBIM 数据集：负责数据加载、校验并调用 transform 完成原有增强。
from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from s3dis_sam3d import decode_semantic_labels
from torch.utils.data import DataLoader, Dataset

from .transform import (
    Compose,
    RandomBIMEdgeDilation,
    RandomBIMFullDropout,
    RandomBIMLogNoise,
    RandomBIMShift,
    RandomBIMSquareDropout,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRGBGainBias,
)


def to_semancit_path(rgb_path):
    semantic_path = (
    rgb_path.parent.parent
    / "semantic"
    / rgb_path.name.replace("_rgb.png", "_semantic.png")
)
    return semantic_path

transform = Compose(
    [
        RandomRGBGainBias(
            gain_range=0.1,
            bias_range=0.1,
            p=1.0,
        ),
        RandomBIMShift(
            max_dx=4,
            max_dy=4,
            p=0.2,
        ),
        RandomBIMSquareDropout(
            fraction=0.12,
            p=0.15,
        ),
        RandomBIMFullDropout(p=0.03),
        RandomBIMLogNoise(
            sigma=0.02,
            p=0.2,
        ),
        RandomBIMEdgeDilation(
            pixels=3,
            p=0.15,
        ),
        RandomHorizontalFlip(
            p=0.5,
            update_intrinsic=False,
            flip_bim_normal_x=True,
        ),
        RandomCrop(
            height=504,
            width=504,
            update_intrinsic=False,
        ),
    ]
)

class S23PriorBIMDataset(Dataset):
    """
    Stanford 2D-3D-S prepared dataset.

    Area_1 is the primary dataset. Extra roots contribute train-only SyncBIM
    samples, so validation and test always keep the original Area_1 protocol.

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

    Optional original PriorBIMDA fields
    -----------------------------------
    bim_normals
    bim_edge

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
        extra_dataset_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
        extra_dataset_stride: int = 1,
    ):
        super().__init__()

        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.extra_dataset_roots = tuple(
            Path(path).expanduser().resolve()
            for path in (extra_dataset_roots or ())
        )
        self.dataset_roots = (self.dataset_root, *self.extra_dataset_roots)
        if len(self.dataset_roots) != len(set(self.dataset_roots)):
            raise ValueError("Dataset roots must be unique")
        self.extra_dataset_stride = int(extra_dataset_stride)
        if self.extra_dataset_stride < 1:
            raise ValueError("extra_dataset_stride must be positive")

        self.s23_root = Path(s23_root).expanduser().resolve()

        self.split = str(split)

        if self.split not in {
            "train",
            "val",
            "test",
        }:
            raise ValueError(f"Unknown split: {self.split}")

        # Default:
        # augment training only.
        if augment is None:
            augment = self.split == "train"

        self.augment = bool(augment)

        if self.augment and self.split != "train":
            raise ValueError("Augmentation is train-only")

        target_shape = (504, 504)
        self.height, self.width = target_shape

        # Keep the same transform order and parameter values as the original
        # PriorBIMDA F36/Adapter training dataset.
        self.transform = transform

        # ---------------------------------------------------------
        # Manifest
        # ---------------------------------------------------------

        roots = self.dataset_roots if self.split == "train" else (self.dataset_root,)
        self.records = []
        for root_index, root in enumerate(roots):
            manifest_path = root / "manifests" / f"{self.split}.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(manifest_path)
            lines = [
                line
                for line in manifest_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if root_index > 0:
                lines = lines[:: self.extra_dataset_stride]
            for line in lines:
                record = json.loads(line)
                if record.get("split") != self.split:
                    raise ValueError(f"{record['id']}: manifest split mismatch")
                record["_dataset_root"] = str(root)
                record.setdefault(
                    "training_source",
                    "real_bim" if root_index == 0 else "syncbim",
                )
                self.records.append(record)

        if not self.records:
            raise ValueError(f"Empty {self.split} dataset in {roots}")

        # ---------------------------------------------------------
        # Basic consistency checks
        # ---------------------------------------------------------

        ids = [str(record["id"]) for record in self.records]

        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in {manifest_path}")

        self.source_counts = {
            source: sum(
                record["training_source"] == source for record in self.records
            )
            for source in sorted(
                {record["training_source"] for record in self.records}
            )
        }

    # =============================================================
    # Paths
    # =============================================================

    def _sample_path(
        self,
        record: dict,
    ) -> Path:
        path = Path(record["sample"])

        if not path.is_absolute():
            path = Path(record["_dataset_root"]) / path

        return path

    def _rgb_path(
        self,
        record: dict,
    ) -> Path:
        path = Path(record["rgb"])

        if not path.is_absolute():
            path = self.s23_root / path

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
            raise RuntimeError(f"Cannot read RGB: {path}")

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        if image.shape[0] != self.height or image.shape[1] != self.width:
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

        rgb = image.astype(np.float32) / 255.0

        # HWC -> CHW
        return np.ascontiguousarray(rgb.transpose(2, 0, 1))

    # =============================================================
    # NPZ
    # =============================================================

    def _load_semantic(
        self,
        path: Path,
    ) -> np.ndarray:
        semantic = decode_semantic_labels(path)

        if semantic.ndim != 2:
            raise ValueError(
                f"Semantic label must be HxW, got {semantic.shape}"
            )

        if semantic.shape != (self.height, self.width):
            # Semantic labels are categorical:
            # nearest-neighbour interpolation only.
            semantic = cv2.resize(
                semantic.astype(np.float32),
                (self.width, self.height),
                interpolation=cv2.INTER_NEAREST_EXACT,
            ).astype(np.int32)

        return np.ascontiguousarray(semantic)

    def _load_sample(
        self,
        path: Path,
    ) -> dict[str, np.ndarray]:

        if not path.is_file():
            raise FileNotFoundError(path)

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

            missing = required - set(item.files)

            if missing:
                raise ValueError(f"{path}: missing keys {sorted(missing)}")

            intrinsic = item["intrinsic"].astype(np.float32)

            da3_raw = item["da3_depth_raw"].astype(np.float32)

            focal_scale = float(item["da3_focal_scale"].item())

            bim_depth = item["bim_depth"].astype(np.float32)

            bim_valid = item["bim_valid"].astype(np.float32)

            gt_depth = item["gt_depth"].astype(np.float32)

            gt_valid = item["gt_valid"].astype(np.float32)

            optional_bim = {}
            if "bim_normals" in item.files:
                optional_bim["bim_normals"] = item["bim_normals"].astype(np.float32)
            if "bim_edge" in item.files:
                optional_bim["bim_edge"] = item["bim_edge"].astype(np.float32)

        # ---------------------------------------------------------
        # DA3 metric focal correction
        # ---------------------------------------------------------

        if not np.isfinite(focal_scale) or focal_scale <= 0:
            raise ValueError(f"{path}: invalid da3_focal_scale={focal_scale}")

        da3_depth = da3_raw * focal_scale

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
                    f"{path}: {name}.shape={value.shape}, expected={shape}"
                )

        if intrinsic.shape != (3, 3):
            raise ValueError(f"{path}: intrinsic shape={intrinsic.shape}")

        if "bim_normals" in optional_bim and optional_bim["bim_normals"].shape != (
            3,
            *shape,
        ):
            raise ValueError(
                f"{path}: bim_normals.shape={optional_bim['bim_normals'].shape}, "
                f"expected={(3, *shape)}"
            )
        if "bim_edge" in optional_bim and optional_bim["bim_edge"].shape != shape:
            raise ValueError(
                f"{path}: bim_edge.shape={optional_bim['bim_edge'].shape}, "
                f"expected={shape}"
            )

        if not np.isfinite(da3_depth).all() or np.any(da3_depth <= 0):
            raise ValueError(f"{path}: invalid DA3 depth")

        bim_valid = bim_valid > 0

        gt_valid = gt_valid > 0

        if np.any(bim_valid & (~np.isfinite(bim_depth) | (bim_depth <= 0))):
            raise ValueError(f"{path}: valid BIM pixels contain bad depth")

        if np.any((~bim_valid) & (bim_depth != 0)):
            raise ValueError(f"{path}: invalid BIM pixels are not zero")

        if np.any(gt_valid & (~np.isfinite(gt_depth) | (gt_depth <= 0))):
            raise ValueError(f"{path}: valid GT pixels contain bad depth")

        # Return [C,H,W].
        arrays = {
            "intrinsic": intrinsic,
            "da3_depth": da3_depth[None],
            "bim_depth": bim_depth[None],
            "bim_valid": bim_valid.astype(np.float32)[None],
            "gt_depth": gt_depth[None],
            "gt_valid": gt_valid.astype(np.float32)[None],
        }
        if "bim_normals" in optional_bim:
            arrays["bim_normals"] = optional_bim["bim_normals"]
        if "bim_edge" in optional_bim:
            arrays["bim_edge"] = optional_bim["bim_edge"][None]
        return arrays

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

        sample_path = self._sample_path(record)

        rgb_path = self._rgb_path(record)
        semantic_path = to_semancit_path(rgb_path)
        semantic_labels = self._load_semantic(semantic_path)

        arrays = self._load_sample(sample_path)

        arrays["rgb"] = self._load_rgb(rgb_path)
        arrays["semantic_labels"] = semantic_labels

        if self.augment:
            arrays = self._augment(arrays)

        # ---------------------------------------------------------
        # numpy -> torch
        # ---------------------------------------------------------

        output = {
            key: torch.from_numpy(np.ascontiguousarray(value))
            for key, value in arrays.items()
        }

        # ---------------------------------------------------------
        # Metadata stays Python-native.
        # ---------------------------------------------------------

        output.update(
            {
                "sample_id": str(record["id"]),
                "region": str(record["region"]),
                "area": str(
                    record.get("area", Path(record["rgb"]).parts[0])
                ),
                "training_source": str(record["training_source"]),
                "camera_uuid": str(record["camera_uuid"]),
                "frame_number": int(record["frame_number"]),
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
    extra_dataset_roots=None,
    extra_dataset_stride=1,
    pin_memory=True,
    persistent_workers=True,
    drop_last=None,
):
    dataset = S23PriorBIMDataset(
        dataset_root=dataset_root,
        s23_root=s23_root,
        split=split,
        augment=augment,
        extra_dataset_roots=extra_dataset_roots,
        extra_dataset_stride=extra_dataset_stride,
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
        persistent_workers=(persistent_workers),
        drop_last=drop_last,
    )

    return loader
