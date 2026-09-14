import json
from pathlib import Path

from data.dataset import S23PriorBIMDataset


def _write_manifest(root: Path, split: str, records):
    path = root / "manifests" / f"{split}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _record(sample_id, split, region, sample):
    return {
        "id": sample_id,
        "split": split,
        "region": region,
        "rgb": f"{sample_id}_rgb.png",
        "sample": sample,
        "camera_uuid": "camera",
        "frame_number": 0,
    }


def test_extra_roots_only_expand_training_split(tmp_path: Path):
    primary = tmp_path / "area1"
    syncbim = tmp_path / "syncbim"
    _write_manifest(
        primary,
        "train",
        [_record("area_1/office_1/frame_0", "train", "office_1", "samples/a.npz")],
    )
    _write_manifest(
        primary,
        "val",
        [_record("area_1/office_2/frame_0", "val", "office_2", "samples/b.npz")],
    )
    _write_manifest(
        syncbim,
        "train",
        [
            {
                **_record(
                    "area_2/office_1/frame_0",
                    "train",
                    "area_2/office_1",
                    "samples/area_2/a.npz",
                ),
                "area": "area_2",
                "training_source": "syncbim",
            }
        ],
    )

    train = S23PriorBIMDataset(
        primary,
        tmp_path,
        "train",
        augment=False,
        extra_dataset_roots=[syncbim],
    )
    validation = S23PriorBIMDataset(
        primary,
        tmp_path,
        "val",
        augment=False,
        extra_dataset_roots=[syncbim],
    )

    assert len(train) == 2
    assert len(validation) == 1
    assert train.source_counts == {"real_bim": 1, "syncbim": 1}
    assert train._sample_path(train.records[0]) == primary / "samples/a.npz"
    assert train._sample_path(train.records[1]) == syncbim / "samples/area_2/a.npz"


def test_extra_dataset_stride_does_not_sample_primary_root(tmp_path: Path):
    primary = tmp_path / "area1"
    syncbim = tmp_path / "syncbim"
    _write_manifest(
        primary,
        "train",
        [_record("area_1/frame_0", "train", "area_1/office_1", "samples/a.npz")],
    )
    _write_manifest(
        syncbim,
        "train",
        [
            _record(
                f"area_2/frame_{index}",
                "train",
                "area_2/office_1",
                f"samples/{index}.npz",
            )
            for index in range(5)
        ],
    )

    dataset = S23PriorBIMDataset(
        primary,
        tmp_path,
        "train",
        augment=False,
        extra_dataset_roots=[syncbim],
        extra_dataset_stride=2,
    )

    assert [record["id"] for record in dataset.records] == [
        "area_1/frame_0",
        "area_2/frame_0",
        "area_2/frame_2",
        "area_2/frame_4",
    ]
    assert dataset.source_counts == {"real_bim": 1, "syncbim": 3}
