"""ScanNet source labels and metadata adapted to the existing S23 dataset."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from s3dis_sam3d.s23dis import S23DIS_CLASS_TO_ID
from s3dis_sam3d.scannet import SCANNET_SEMANTIC_CLASSES


class ScanNetDatasetAdapter:
    def __init__(self, target_shape=(504, 504)):
        self.target_shape = target_shape
        aliases = {
            "bookshelf": "bookcase",
            "shelves": "bookcase",
            "desk": "table",
            "whiteboard": "board",
        }
        # Unmatched NYU40 classes remain -1; GT validity is not affected.
        self.label_lookup = np.array(
            [
                S23DIS_CLASS_TO_ID.get(aliases.get(name, name), -1)
                for name in SCANNET_SEMANTIC_CLASSES
            ],
            dtype=np.int32,
        )
        # Weight groups, not semantic aliases: e.g. a bed is not a S23 sofa.
        furniture = {
            "cabinet", "bed", "chair", "sofa", "table", "bookshelf",
            "counter", "desk", "shelves", "dresser", "night stand", "otherfurniture",
        }
        self.furniture_lookup = np.array(
            [name in furniture for name in SCANNET_SEMANTIC_CLASSES], dtype=bool
        )

    def load_semantics(self, rgb_path):
        rgb_path = Path(rgb_path)
        path = rgb_path.parent.parent / "label" / f"{rgb_path.stem}.png"
        labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if labels is None:
            raise RuntimeError(f"Cannot read ScanNet semantic: {path}")
        if labels.ndim != 2:
            raise ValueError(f"ScanNet labels must be HxW, got {labels.shape}: {path}")

        if labels.shape != self.target_shape:
            height, width = self.target_shape
            labels = cv2.resize(
                labels.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST_EXACT,
            ).astype(np.int32)
        valid = (labels >= 0) & (labels < len(self.label_lookup))
        semantic = np.full(labels.shape, -1, dtype=np.int32)
        furniture_mask = np.zeros(labels.shape, dtype=np.float32)
        semantic[valid] = self.label_lookup[labels[valid]]
        furniture_mask[valid] = self.furniture_lookup[labels[valid]]
        return {
            "semantic_labels": np.ascontiguousarray(semantic),
            "furniture_mask": np.ascontiguousarray(furniture_mask[None]),
        }

    def load_semantic(self, rgb_path):
        return self.load_semantics(rgb_path)["semantic_labels"]

    def camera_uuid(self, record):
        # ScanNet has scene/frame IDs rather than S23 camera UUIDs.
        return str(record.get("camera_uuid", record["region"]))
