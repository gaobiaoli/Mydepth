import json

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data.dataset import S23PriorBIMDataset, to_semancit_path
from data.scannet_adapter import ScanNetDatasetAdapter
from data.transform import Compose, RandomCrop, RandomHorizontalFlip, Resize
from loss import depth_weights


def _write_labels(rgb_path, labels):
    path = rgb_path.parent.parent / "label" / f"{rgb_path.stem}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), labels)


def test_scannet_labels_map_to_s23_and_weight_groups(tmp_path):
    rgb_path = tmp_path / "scene0000_00/color/000000.jpg"
    labels = np.array([list(range(41)) + [255]], dtype=np.uint16)
    _write_labels(rgb_path, labels)
    adapter = ScanNetDatasetAdapter(labels.shape)

    expected = np.full(labels.shape, -1, dtype=np.int32)
    # Genuine counterparts/aliases; beds and cabinets keep unmatched semantics.
    mapping = {1: 2, 2: 1, 5: 8, 6: 9, 7: 7, 8: 6, 9: 5, 10: 10, 14: 7, 15: 10, 22: 0, 30: 11}
    for source, target in mapping.items():
        expected[labels == source] = target
    fields = adapter.load_semantics(rgb_path)
    actual = fields["semantic_labels"]

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int32
    assert actual.flags.c_contiguous
    assert set(labels[np.isin(actual, [7, 8, 9, 10])]) == {5, 6, 7, 10, 14, 15}
    furniture_ids = {3, 4, 5, 6, 7, 10, 12, 14, 15, 17, 32, 39}
    mask = fields["furniture_mask"]
    assert mask.shape == (1, *labels.shape)
    assert mask.dtype == np.float32 and mask.flags.c_contiguous
    assert set(labels[mask[0] > 0]) == furniture_ids
    np.testing.assert_array_equal(adapter.load_semantic(rgb_path), actual)


def test_scannet_label_resize_preserves_categories(tmp_path):
    rgb_path = tmp_path / "scene0000_00/color/000000.jpg"
    _write_labels(rgb_path, np.array([[5, 8], [0, 10]], dtype=np.uint8))

    labels = ScanNetDatasetAdapter((4, 6)).load_semantic(rgb_path)

    expected = np.repeat(np.repeat([[8, 6], [-1, 10]], 2, axis=0), 3, axis=1)
    np.testing.assert_array_equal(labels, expected)


def test_scannet_extended_furniture_gets_original_additive_weights(tmp_path):
    rgb_path = tmp_path / "scene0000_00/color/000000.png"
    ids = [3, 4, 5, 6, 7, 10, 12, 14, 15, 17, 32, 39, 1, 24, 33, 36, 0, 255]
    labels = np.array([ids], dtype=np.uint8)
    _write_labels(rgb_path, labels)
    fields = ScanNetDatasetAdapter(labels.shape).load_semantics(rgb_path)
    semantic = torch.from_numpy(fields["semantic_labels"])[None]
    mask = torch.from_numpy(fields["furniture_mask"])[None]
    depth = torch.full_like(mask, 2)
    batch = {
        "semantic_labels": semantic,
        "furniture_mask": mask,
        "gt_depth": depth,
        "gt_valid": torch.ones_like(mask),
        "bim_depth": depth.clone(),
        "bim_valid": torch.ones_like(mask),
    }
    expected = torch.tensor([2] * 12 + [1] * 6, dtype=torch.float32)
    torch.testing.assert_close(depth_weights(batch).flatten(), expected)
    # Near-depth/conflict bonuses remain additive; furniture is not counted twice.
    batch["gt_depth"] = torch.full_like(mask, 0.8)
    torch.testing.assert_close(depth_weights(batch).flatten(), expected + 3)
    batch["gt_valid"][..., 1] = 0  # Bed still has furniture semantics, but invalid GT.
    assert depth_weights(batch)[..., 1] == 0


def test_explicit_s23_furniture_mask_keeps_original_weights():
    labels = torch.arange(-1, 13).view(1, 1, 14)
    depth = torch.full((1, 1, 1, 14), 0.8)
    batch = {
        "semantic_labels": labels,
        "gt_depth": depth,
        "gt_valid": torch.ones_like(depth),
        "bim_depth": torch.full_like(depth, 2),
        "bim_valid": torch.ones_like(depth),
    }
    expected = depth_weights(batch)
    batch["furniture_mask"] = torch.isin(labels, torch.tensor([7, 8, 9, 10])).float()[:, None]
    torch.testing.assert_close(depth_weights(batch), expected)


def test_furniture_mask_follows_crop_and_nearest_resize():
    import random

    mask = np.zeros((1, 6, 8), dtype=np.float32)
    mask[:, 1:5, 2:6] = 1
    sample = {"rgb": np.repeat(mask, 3, axis=0), "furniture_mask": mask}
    result = RandomCrop(4, 4, update_intrinsic=False)(sample, rng=random.Random(42))
    np.testing.assert_array_equal(result["rgb"][0], result["furniture_mask"][0])
    expected = cv2.resize(result["furniture_mask"][0], (8, 8), interpolation=cv2.INTER_NEAREST)
    result = Resize(8, 8)(result, rng=random.Random(42))
    np.testing.assert_array_equal(result["furniture_mask"][0], expected)
    assert set(np.unique(result["furniture_mask"])) <= {0, 1}


def test_missing_scannet_labels_are_not_silently_ignored(tmp_path):
    rgb_path = tmp_path / "scene0000_00/color/000000.jpg"
    with pytest.raises(RuntimeError, match="Cannot read ScanNet semantic"):
        ScanNetDatasetAdapter().load_semantic(rgb_path)


def test_scannet_camera_uuid_preserves_value_or_uses_region():
    adapter = ScanNetDatasetAdapter()
    assert (
        adapter.camera_uuid({"region": "scannet/scene0000_00"})
        == "scannet/scene0000_00"
    )
    assert (
        adapter.camera_uuid({"region": "scannet/scene0000_00", "camera_uuid": "camera"})
        == "camera"
    )


def _write_sample(root, record, rgb_path):
    manifest = root / "manifests/train.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rgb_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(rgb_path), np.full((2, 2, 3), 128, dtype=np.uint8))

    sample = root / record["sample"]
    sample.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full((504, 504), 2, dtype=np.float32)
    valid = np.ones((504, 504), dtype=np.uint8)
    np.savez(
        sample,
        sample_schema_version=np.array(1, dtype=np.uint16),
        intrinsic=np.eye(3, dtype=np.float32),
        da3_depth_raw=depth / 2,
        da3_focal_scale=np.array(2, dtype=np.float32),
        bim_depth=depth,
        bim_valid=valid,
        gt_depth=depth,
        gt_valid=valid,
    )


@pytest.fixture
def mixed_dataset(tmp_path, monkeypatch):
    primary = tmp_path / "area1"
    extra = tmp_path / "scannet"
    source = tmp_path / "source"
    s23_rgb = source / "area_1/data/rgb/frame_domain_rgb.png"
    scannet_rgb = source / "scene0000_00/color/000000.jpg"
    record = {
        "id": "area_1/office_1/frame_0",
        "split": "train",
        "region": "office_1",
        "area": "area_1",
        "rgb": str(s23_rgb.relative_to(source)),
        "sample": "samples/frame_0.npz",
        "camera_uuid": "camera",
        "frame_number": 0,
    }
    _write_sample(primary, record, s23_rgb)
    _write_sample(
        extra,
        {
            **{k: v for k, v in record.items() if k != "camera_uuid"},
            "id": "scannet/scene0000_00/000000",
            "region": "scannet/scene0000_00",
            "area": "scannet",
            "dataset": "scannet",
            "rgb": str(scannet_rgb),
            "training_source": "syncbim",
        },
        scannet_rgb,
    )
    _write_labels(scannet_rgb, np.array([[5, 8], [0, 7]], dtype=np.uint8))

    def load_s23_semantic(self, path):
        # The S23 decoder must not receive ScanNet label images.
        assert path == to_semancit_path(s23_rgb)
        return np.full((504, 504), 2, dtype=np.int32)

    monkeypatch.setattr(S23PriorBIMDataset, "_load_semantic", load_s23_semantic)
    return S23PriorBIMDataset(
        primary, source, "train", augment=False, extra_dataset_roots=[extra]
    )


def test_mixed_batch_keeps_fields_and_original_loss_weights(mixed_dataset):
    batch = next(iter(DataLoader(mixed_dataset, batch_size=2, num_workers=0)))

    assert batch["rgb"].shape == (2, 3, 504, 504)
    assert batch["gt_depth"].shape == (2, 1, 504, 504)
    assert batch["semantic_labels"].shape == (2, 504, 504)
    assert batch["semantic_labels"].dtype == torch.int32
    assert batch["furniture_mask"].shape == (2, 1, 504, 504)
    assert batch["camera_uuid"] == ["camera", "scannet/scene0000_00"]
    assert batch["training_source"] == ["real_bim", "syncbim"]
    assert torch.equal(batch["da3_depth"], batch["gt_depth"])

    weights = depth_weights(batch)
    assert torch.all(weights[0] == 1)  # S23 wall
    assert weights[1, 0, 0, 0] == 2  # ScanNet chair
    assert weights[1, 0, 0, -1] == 1  # ScanNet door, not furniture
    assert weights[1, 0, -1, 0] == 1  # Unannotated semantics still have valid GT
    assert weights[1, 0, -1, -1] == 2  # ScanNet table


def test_scannet_semantics_follow_existing_horizontal_flip(mixed_dataset):
    original = mixed_dataset[1]
    mixed_dataset.augment = True
    mixed_dataset.transform = Compose(
        [RandomHorizontalFlip(p=1, update_intrinsic=False)]
    )
    flipped = mixed_dataset[1]

    assert torch.equal(flipped["semantic_labels"], original["semantic_labels"].flip(-1))
    assert torch.equal(flipped["furniture_mask"], original["furniture_mask"].flip(-1))
    assert torch.equal(flipped["rgb"], original["rgb"].flip(-1))
    assert torch.equal(flipped["gt_valid"], original["gt_valid"])
