"""Tests for lowlight.datasets (reading files, pairing, subsets, checks)."""

import numpy as np
import pytest
from conftest import FAKE_COUNTS

from lowlight.datasets import (
    DatasetNotFoundError,
    Sample,
    build_dataset,
    check_datasets,
    detect_label_format,
    find_acdc,
    find_cityscapes,
    find_darkzurich,
    find_folder,
    find_nightcity,
    format_check_report,
    missing_labels,
    read_image,
    read_label,
    select_subset,
    write_image,
    write_label,
)

# ---------- reading and writing ----------


def test_read_image_returns_rgb_not_bgr(tmp_path):
    red = np.zeros((4, 5, 3), dtype=np.uint8)
    red[..., 0] = 255  # pure red in RGB order
    write_image(tmp_path / "red.png", red)
    back = read_image(tmp_path / "red.png")
    assert back.dtype == np.uint8 and back.shape == (4, 5, 3)
    np.testing.assert_array_equal(back, red)


def test_read_image_converts_gray_and_rgba(tmp_path):
    write_label(tmp_path / "gray.png", np.full((3, 4), 77, dtype=np.uint8))  # 1-channel file
    gray = read_image(tmp_path / "gray.png")
    assert gray.shape == (3, 4, 3) and (gray == 77).all()
    bgra = np.zeros((3, 4, 4), dtype=np.uint8)
    bgra[..., 2] = 200  # red (OpenCV stores BGRA, so index 2 is red)
    bgra[..., 3] = 128  # half-transparent alpha channel
    write_label(tmp_path / "rgba.png", bgra)  # write_label saves the array exactly as given
    rgb = read_image(tmp_path / "rgba.png")
    assert rgb.shape == (3, 4, 3) and (rgb[..., 0] == 200).all()


def test_read_image_works_with_non_english_path(tmp_path):
    path = tmp_path / "छवि" / "रात.png"  # Hindi folder and file names
    write_image(path, np.full((2, 2, 3), 9, dtype=np.uint8))
    assert read_image(path).shape == (2, 2, 3)


def test_read_image_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "nope.png")
    (tmp_path / "bad.png").write_bytes(b"this is not an image")
    with pytest.raises(ValueError, match="decode"):
        read_image(tmp_path / "bad.png")


def test_read_label_rejects_colour_label(tmp_path):
    write_image(tmp_path / "color.png", np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="single-channel"):
        read_label(tmp_path / "color.png")


# ---------- finding pairs ----------


def test_find_acdc_pairs_and_ids(fake_data):
    train = find_acdc(fake_data["acdc_root"], "train")
    assert [s.image_id for s in train] == [
        "GOPR0001_frame_000001",
        "GOPR0001_frame_000002",
        "GOPR0002_frame_000001",
    ]
    assert all(s.label_path.is_file() for s in train)
    assert train[0].label_path.name == "GOPR0001_frame_000001_gt_labelTrainIds.png"
    assert len(find_acdc(fake_data["acdc_root"], "val")) == 2


def test_acdc_test_split_is_refused(fake_data):
    with pytest.raises(ValueError, match="no labels"):
        find_acdc(fake_data["acdc_root"], "test")


def test_missing_folder_raises(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        find_cityscapes(tmp_path / "nothing")


def test_cityscapes_labels_converted_to_trainids(fake_data):
    samples = find_cityscapes(fake_data["cityscapes_root"])
    assert len(samples) == 2 and samples[0].label_format == "labelid"
    mask = read_label(samples[0].label_path, samples[0].label_format)
    assert mask[0, 0] == 10  # labelId 23 (sky) -> trainId 10
    assert mask[5, 5] == 0  # labelId 7 (road) -> trainId 0
    assert mask[3, 3] == 13  # labelId 26 (car) -> trainId 13
    assert mask[-1, -1] == 255  # labelId 0 (unlabeled) -> ignore


def test_find_darkzurich(fake_data):
    assert [s.image_id for s in find_darkzurich(fake_data["darkzurich_root"])] == [
        "GOPR0356_frame_000321"
    ]


def test_find_nightcity_with_unknown_layout(fake_data):
    train = find_nightcity(fake_data["nightcity_root"], "train")
    assert [s.image_id for s in train] == ["Chicago_0001", "Tokyo_0002"]
    assert all("NightCity-images" in str(s.image_path) for s in train)  # images, not labels
    assert all("label" in str(s.label_path) and "_color" not in str(s.label_path) for s in train)
    assert train[0].label_format == "labelid"  # detected from the values, not the name
    mask = read_label(train[0].label_path, train[0].label_format)
    assert mask[0, 0] == 10 and mask[3, 3] == 13  # sky and car as trainIds
    val = find_nightcity(fake_data["nightcity_root"], "val")  # stored in a folder called "test"
    assert [s.image_id for s in val] == ["Berlin_0003"]


def test_find_paired_tree_reports_image_without_label(fake_data):
    root = fake_data["nightcity_root"]
    write_image(
        root / "NightCity-images/images/train/Extra_0009.png", np.zeros((8, 12, 3), np.uint8)
    )
    junk = root / "__MACOSX/NightCity-images/images/train/._Chicago_0001.png"
    junk.parent.mkdir(parents=True)
    junk.write_bytes(b"mac metadata, not an image")
    samples = find_nightcity(root, "train")
    assert [s.image_id for s in missing_labels(samples)] == ["Extra_0009"]
    assert not any("__MACOSX" in str(s.image_path) for s in samples)


def test_detect_label_format(tmp_path):
    trainid = np.zeros((4, 4), np.uint8)
    trainid[0] = 255
    write_label(tmp_path / "a.png", trainid)
    labelid = np.full((4, 4), 7, np.uint8)
    labelid[0] = 23
    write_label(tmp_path / "b.png", labelid)
    assert detect_label_format([tmp_path / "a.png"]) == "trainid"
    assert detect_label_format([tmp_path / "b.png"]) == "labelid"
    assert detect_label_format([tmp_path / "x_labelTrainIds.png"]) == "trainid"  # name decides


def test_missing_labels_detected(fake_data):
    samples = find_acdc(fake_data["acdc_root"], "val")
    samples[0].label_path.unlink()
    assert missing_labels(samples) == [samples[0]]


def test_find_folder_with_optional_labels(tmp_path):
    write_image(tmp_path / "imgs" / "a.jpg", np.zeros((2, 2, 3), dtype=np.uint8))
    write_image(tmp_path / "imgs" / "b.png", np.zeros((2, 2, 3), dtype=np.uint8))
    (tmp_path / "imgs" / "notes.txt").write_text("not an image")
    write_label(tmp_path / "labels" / "b.png", np.zeros((2, 2), dtype=np.uint8))
    samples = find_folder(tmp_path / "imgs", tmp_path / "labels")
    assert [(s.image_id, s.label_path is not None) for s in samples] == [("a", False), ("b", True)]


# ---------- subsets ----------


def _samples(n):
    return [Sample(f"img{i:03d}", f"img{i:03d}.png") for i in range(n)]


def test_select_subset_is_reproducible_and_sorted():
    samples = _samples(100)
    a = select_subset(samples, 20, seed=42)
    b = select_subset(samples, 20, seed=42)
    assert a == b and len(a) == 20
    assert [s.image_id for s in a] == sorted(s.image_id for s in a)
    assert select_subset(samples, 20, seed=7) != a  # a different seed gives a different subset


def test_select_subset_without_limit_returns_all():
    samples = _samples(5)
    assert select_subset(samples, None, seed=1) == samples
    assert select_subset(samples, 50, seed=1) == samples


# ---------- dataset object ----------


def test_build_dataset_loads_items(fake_cfg):
    ds = build_dataset(fake_cfg, "acdc_night", "train")
    assert len(ds) == 3
    image, label, sample = ds[0]
    assert image.dtype == np.uint8 and image.shape == (8, 12, 3)
    assert label.dtype == np.uint8 and label.shape == (8, 12)
    assert sample.image_id == "GOPR0001_frame_000001"
    assert len(list(ds)) == 3


def test_build_dataset_applies_limit(fake_cfg):
    fake_cfg["limit"] = 2
    ds = build_dataset(fake_cfg, "acdc_night", "train")
    assert len(ds) == 2
    assert ds.image_ids == build_dataset(fake_cfg, "acdc_night", "train").image_ids


def test_build_dataset_errors(fake_cfg, fake_data):
    with pytest.raises(KeyError):
        build_dataset(fake_cfg, "kitti", "val")
    with pytest.raises(ValueError, match="split"):
        build_dataset(fake_cfg, "cityscapes", "train")
    find_acdc(fake_data["acdc_root"], "val")[0].label_path.unlink()
    with pytest.raises(ValueError, match="labels missing"):
        build_dataset(fake_cfg, "acdc_night", "val")


# ---------- check-data ----------


def test_check_datasets_all_ok(fake_cfg):
    results = check_datasets(fake_cfg, deep=True, expected_counts=FAKE_COUNTS)
    assert [(r.name, r.split, r.found, r.status) for r in results] == [
        ("acdc_night", "train", 3, "ok"),
        ("acdc_night", "val", 2, "ok"),
        ("cityscapes", "val", 2, "ok"),
        ("darkzurich", "val", 1, "ok"),
        ("nightcity", "train", 2, "ok"),
        ("nightcity", "val", 1, "ok"),
    ]
    assert "ALL REQUIRED DATA OK" in format_check_report(results)


def test_check_reports_wrong_count_and_size_mismatch(fake_cfg, fake_data):
    label = find_acdc(fake_data["acdc_root"], "val")[1].label_path
    write_label(label, np.zeros((3, 3), dtype=np.uint8))  # wrong size
    results = check_datasets(fake_cfg, names=["acdc_night"], deep=True)  # real counts: 400/106
    assert all(r.status == "problem" for r in results)
    text = format_check_report(results)
    assert "expected 400" in text and "vs label (3, 3)" in text
    assert "DATA CHECK FAILED" in text


def test_check_flags_bad_label_values(fake_cfg, fake_data):
    label = find_acdc(fake_data["acdc_root"], "train")[0].label_path
    write_label(label, np.full((8, 12), 42, dtype=np.uint8))  # 42 is not a trainId
    (result,) = [
        r for r in check_datasets(fake_cfg, ["acdc_night"], True, FAKE_COUNTS) if r.split == "train"
    ]
    assert any("unexpected label values [42]" in p for p in result.problems)


def test_optional_dataset_missing_does_not_fail(fake_cfg, fake_data, tmp_path):
    fake_cfg["datasets"]["darkzurich_root"] = str(tmp_path / "not_downloaded")
    results = check_datasets(fake_cfg, expected_counts=FAKE_COUNTS)
    (dz,) = [r for r in results if r.name == "darkzurich"]
    assert dz.status == "missing" and dz.passed
    assert "ALL REQUIRED DATA OK" in format_check_report(results)


def test_required_dataset_missing_fails(fake_cfg, tmp_path):
    fake_cfg["datasets"]["cityscapes_root"] = str(tmp_path / "not_downloaded")
    results = check_datasets(fake_cfg, names=["cityscapes"], expected_counts=FAKE_COUNTS)
    assert results[0].status == "missing" and not results[0].passed
