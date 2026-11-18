"""Tests for lowlight.metrics with a hand-checked 4x4 example (PID Section 11.3)."""

import numpy as np
import pytest

from lowlight.metrics import confusion_matrix, image_scores, load_cm, metrics_from_cm, save_cm

# 3 classes, 255 = ignore. Worked out by hand:
#   class 0: 5 true pixels, 4 predicted right, 1 predicted as class 1
#   class 1: 4 true pixels, 3 right, 1 predicted as class 0
#   class 2: 5 true pixels, 4 right, 1 predicted as class 1
#   two ignored pixels (they are predicted 0 and 1, which must NOT count)
GT = np.array(
    [[0, 0, 1, 1],
     [0, 0, 1, 1],
     [2, 2, 255, 255],
     [2, 2, 2, 0]], dtype=np.uint8)  # fmt: skip
PRED = np.array(
    [[0, 0, 1, 0],
     [0, 1, 1, 1],
     [2, 2, 0, 1],
     [2, 1, 2, 0]], dtype=np.uint8)  # fmt: skip
EXPECTED_CM = np.array([[4, 1, 0], [1, 3, 0], [0, 1, 4]])


def test_confusion_matrix_hand_example():
    cm = confusion_matrix(PRED, GT, num_classes=3)
    np.testing.assert_array_equal(cm, EXPECTED_CM)
    assert cm.sum() == 14  # 16 pixels minus 2 ignored


def test_metrics_hand_example():
    m = metrics_from_cm(EXPECTED_CM)
    # IoU = TP / (TP + FP + FN): class 0: 4/(4+1+1), class 1: 3/(3+2+1), class 2: 4/(4+0+1)
    np.testing.assert_allclose(m["iou"], [4 / 6, 3 / 6, 4 / 5])
    assert m["miou"] == pytest.approx((4 / 6 + 3 / 6 + 4 / 5) / 3)
    # Dice = 2TP / (2TP + FP + FN)
    np.testing.assert_allclose(m["dice"], [8 / 10, 6 / 9, 8 / 9])
    assert m["pixel_acc"] == pytest.approx(11 / 14)
    np.testing.assert_allclose(m["class_acc"], [4 / 5, 3 / 4, 4 / 5])


def test_absent_class_is_nan_and_left_out():
    cm4 = np.zeros((4, 4), dtype=np.int64)
    cm4[:3, :3] = EXPECTED_CM  # class 3: never true, never predicted
    m = metrics_from_cm(cm4)
    assert np.isnan(m["iou"][3]) and np.isnan(m["dice"][3])
    assert m["miou"] == pytest.approx(metrics_from_cm(EXPECTED_CM)["miou"])


def test_false_alarm_class_gets_zero_iou():
    gt = np.zeros((2, 2), np.uint8)
    pred = np.array([[0, 0], [0, 1]], np.uint8)  # class 1 predicted but not in ground truth
    m = metrics_from_cm(confusion_matrix(pred, gt, num_classes=2))
    assert m["iou"][1] == 0 and m["miou"] == pytest.approx((3 / 4 + 0) / 2)


def test_all_ignored_image_gives_nan_not_crash():
    gt = np.full((3, 3), 255, np.uint8)
    cm, scores = image_scores(np.zeros((3, 3), np.uint8), gt)
    assert cm.sum() == 0 and np.isnan(scores["miou"]) and np.isnan(scores["pixel_acc"])


def test_perfect_prediction():
    gt = np.random.default_rng(0).integers(0, 19, (20, 30)).astype(np.uint8)
    _, scores = image_scores(gt, gt)
    assert scores["miou"] == 1.0 and scores["pixel_acc"] == 1.0 and scores["mean_dice"] == 1.0


def test_errors():
    with pytest.raises(ValueError, match="shape"):
        confusion_matrix(np.zeros((2, 2), np.uint8), np.zeros((3, 3), np.uint8))
    with pytest.raises(ValueError, match="< 3"):
        confusion_matrix(np.full((2, 2), 7, np.uint8), np.zeros((2, 2), np.uint8), num_classes=3)


def test_dataset_metrics_use_summed_matrices():
    # Image A: big, perfectly right. Image B: tiny, completely wrong.
    a = confusion_matrix(np.zeros((10, 10), np.uint8), np.zeros((10, 10), np.uint8), 2)
    b = confusion_matrix(np.ones((1, 1), np.uint8), np.zeros((1, 1), np.uint8), 2)
    summed = metrics_from_cm(a + b)["pixel_acc"]
    assert summed == pytest.approx(100 / 101)  # not the average of per-image (1.0 + 0.0) / 2


def test_save_and_load(tmp_path):
    save_cm(tmp_path / "cm" / "img1.npz", EXPECTED_CM)
    np.testing.assert_array_equal(load_cm(tmp_path / "cm" / "img1.npz"), EXPECTED_CM)
