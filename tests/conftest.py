"""Shared test fixtures: tiny fake datasets with the real folder layout (no downloads needed)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from lowlight.config import load_config
from lowlight.datasets import write_image, write_label

H, W = 8, 12  # tiny images keep the tests fast


def _img(value: int) -> np.ndarray:
    """A small RGB image whose pixels encode `value`, so different files differ."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[..., 0] = value
    img[..., 1] = 255 - value
    return img


def _trainid_mask() -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)  # road
    mask[:2] = 10  # sky
    mask[-1, -1] = 255  # one ignored pixel
    return mask


def make_fake_data(root: Path) -> dict[str, Path]:
    """Create fake ACDC night (train 3, val 2), Cityscapes val (2) and Dark Zurich val (1)."""
    acdc, cs, dz = root / "acdc", root / "cityscapes", root / "darkzurich"
    acdc_items = [
        ("train", "GOPR0001", "GOPR0001_frame_000001"),
        ("train", "GOPR0001", "GOPR0001_frame_000002"),
        ("train", "GOPR0002", "GOPR0002_frame_000001"),
        ("val", "GOPR0003", "GOPR0003_frame_000001"),
        ("val", "GOPR0003", "GOPR0003_frame_000002"),
    ]
    for i, (split, seq, stem) in enumerate(acdc_items):
        write_image(acdc / "rgb_anon/night" / split / seq / f"{stem}_rgb_anon.png", _img(10 * i))
        write_label(
            acdc / "gt/night" / split / seq / f"{stem}_gt_labelTrainIds.png", _trainid_mask()
        )

    # Cityscapes labels use raw labelIds: 7 = road, 23 = sky, 26 = car, 0 = unlabeled.
    cs_mask = np.full((H, W), 7, dtype=np.uint8)
    cs_mask[:2] = 23
    cs_mask[3, 3] = 26
    cs_mask[-1, -1] = 0
    for city, stem in [
        ("frankfurt", "frankfurt_000000_000294"),
        ("lindau", "lindau_000000_000019"),
    ]:
        write_image(cs / "leftImg8bit/val" / city / f"{stem}_leftImg8bit.png", _img(50))
        write_label(cs / "gtFine/val" / city / f"{stem}_gtFine_labelIds.png", cs_mask)

    stem = "GOPR0356_frame_000321"
    write_image(dz / "rgb_anon/val/night/GOPR0356" / f"{stem}_rgb_anon.png", _img(5))
    write_label(dz / "gt/val/night/GOPR0356" / f"{stem}_gt_labelTrainIds.png", _trainid_mask())

    # NightCity: layout unknown in advance, so the fake one is deliberately "messy": images and
    # labels come from two different zips, val is stored as "test", labels use labelIds without
    # saying so in the name, and colour previews sit next to them.
    nc = root / "nightcity"
    for split, name in [
        ("train", "Chicago_0001"),
        ("train", "Tokyo_0002"),
        ("test", "Berlin_0003"),
    ]:
        write_image(nc / "NightCity-images/images" / split / f"{name}.png", _img(80))
        write_label(nc / "nightcity-fine/label" / split / f"{name}.png", cs_mask)
        write_image(nc / "nightcity-fine/label" / split / f"{name}_color.png", _img(1))
    return {"acdc_root": acdc, "cityscapes_root": cs, "darkzurich_root": dz, "nightcity_root": nc}


FAKE_COUNTS = {
    ("acdc_night", "train"): 3,
    ("acdc_night", "val"): 2,
    ("cityscapes", "val"): 2,
    ("darkzurich", "val"): 1,
    ("nightcity", "train"): 2,
    ("nightcity", "val"): 1,
}


@pytest.fixture
def scene() -> np.ndarray:
    """A deterministic 64x96 RGB test image with a brightness gradient, shapes and texture."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:96]
    base = (40 + 150 * xx / 95).astype(np.float64)  # left dark, right bright
    img = np.stack([base, base * 0.9, base * 0.8], axis=-1)
    img[10:30, 10:40] = (200, 40, 40)  # a red "car"
    img[40:60, 60:90] = (30, 30, 120)  # a dark blue "building"
    img += rng.normal(0, 4, img.shape)  # mild texture
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture
def fake_data(tmp_path: Path) -> dict[str, Path]:
    return make_fake_data(tmp_path / "data")


@pytest.fixture
def fake_cfg_path(tmp_path: Path, fake_data: dict[str, Path]) -> Path:
    """An experiment YAML pointing the dataset roots at the fake data."""
    path = tmp_path / "fake.yaml"
    roots = {k: str(v) for k, v in fake_data.items()}
    path.write_text(yaml.safe_dump({"datasets": roots}), encoding="utf-8")
    return path


@pytest.fixture
def fake_cfg(fake_cfg_path: Path) -> dict:
    return load_config(fake_cfg_path)
