"""Tests for lowlight.evaluate with a fake model on the fake datasets."""

import numpy as np
import pytest

from lowlight.datasets import build_dataset
from lowlight.evaluate import evaluate
from lowlight.models import DummySegmenter


def _perfect_for_fake_acdc(img):
    """The fake ACDC labels are road everywhere, sky in the top 2 rows, one ignored pixel."""
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[:2] = 10
    return mask


def test_perfect_model_scores_one(fake_cfg):
    ds = build_dataset(fake_cfg, "acdc_night", "train")
    result = evaluate(DummySegmenter(_perfect_for_fake_acdc), ds)
    assert result.summary["miou"] == 1.0 and result.summary["n_images"] == 3
    assert result.cms.shape == (3, 19, 19)
    assert [r["image_id"] for r in result.per_image] == ds.image_ids
    assert all(r["infer_ms"] >= 0 and r["enhance_ms"] >= 0 for r in result.per_image)


def test_wrong_model_and_per_class(fake_cfg):
    ds = build_dataset(fake_cfg, "acdc_night", "val")
    result = evaluate(DummySegmenter(np.zeros((1, 1), np.uint8)), ds)  # always "road"
    iou = result.summary["per_class_iou"]
    assert iou[10] == 0.0  # sky never found
    assert 0 < iou[0] < 1  # road found, plus false alarms where the sky is
    assert np.isnan(iou[13])  # car: neither present nor predicted


def test_darkening_and_enhancement_reach_the_model(fake_cfg):
    seen = []

    def spy(img):
        seen.append(img.mean())
        return _perfect_for_fake_acdc(img)

    ds = build_dataset({**fake_cfg, "limit": 1}, "acdc_night", "train")
    evaluate(DummySegmenter(spy), ds)
    evaluate(DummySegmenter(spy), ds, level="L4")
    evaluate(DummySegmenter(spy), ds, level="L4", method="gamma", params={"gamma": 0.4})
    plain, dark, dark_then_gamma = seen
    assert dark < plain and dark_then_gamma > dark


def test_callback_receives_each_image(fake_cfg):
    calls = []
    ds = build_dataset(fake_cfg, "acdc_night", "val")
    evaluate(DummySegmenter(_perfect_for_fake_acdc), ds, on_image=lambda *a: calls.append(a[0]))
    assert calls == ds.image_ids


def test_needs_labels(fake_cfg, tmp_path):
    from lowlight.datasets import SegDataset, find_folder, write_image

    write_image(tmp_path / "x.png", np.zeros((4, 4, 3), np.uint8))
    ds = SegDataset("folder", "none", find_folder(tmp_path))
    with pytest.raises(ValueError, match="no label"):
        evaluate(DummySegmenter(np.zeros((1, 1), np.uint8)), ds)
