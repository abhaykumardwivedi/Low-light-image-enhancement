"""Segmentation metrics from confusion matrices (PID Section 9.5).

A confusion matrix `cm` has one row per TRUE class and one column per PREDICTED class:
cm[t, p] = number of pixels whose true class is t and predicted class is p. Every metric we report
(IoU, mIoU, Dice, pixel accuracy, class accuracy) can be computed from it, so we save one small
19 x 19 matrix per image and can recompute any metric later without running the models again.

Dataset-level metrics use the SUM of the per-image matrices (standard practice), not the average
of per-image scores: a big object in one image then counts as much as many small ones elsewhere.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from lowlight.labels import IGNORE_INDEX, NUM_CLASSES


def confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Count (true class, predicted class) pairs over all non-ignored pixels.

    Uses one `np.bincount` instead of loops: each valid pixel gets the single number
    gt * C + pred, which is unique for every (gt, pred) pair; counting those numbers and reshaping
    to C x C gives the matrix.

    Returns:
        int64 array of shape (num_classes, num_classes).
    """
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} != ground truth shape {gt.shape}")
    valid = gt != ignore_index
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    if g.size and (g.max() >= num_classes or p.max() >= num_classes):
        raise ValueError(f"Class ids must be < {num_classes} (or {ignore_index} in the label)")
    counts = np.bincount(g * num_classes + p, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def _nanmean(x: np.ndarray) -> float:
    """Mean ignoring NaN; NaN (without a warning) if every value is NaN."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return float(np.nanmean(x)) if x.size else float("nan")


def _safe_divide(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, with NaN wherever den == 0."""
    out = np.full(num.shape, np.nan, dtype=np.float64)
    np.divide(num, den, out=out, where=den > 0)
    return out


def metrics_from_cm(cm: np.ndarray) -> dict[str, np.ndarray | float]:
    """All metrics from one confusion matrix (per image or summed over a dataset).

    For each class c:  TP = cm[c, c],  FP = column sum - TP,  FN = row sum - TP
        IoU  = TP / (TP + FP + FN)
        Dice = 2TP / (2TP + FP + FN)
    A class with no true and no predicted pixels has IoU = Dice = NaN and is left out of the means
    (np.nanmean). A class that is predicted but absent from the ground truth gets IoU 0 (it is a
    pure false alarm), as in the standard Cityscapes evaluation (decision D5).

    Returns:
        dict with per-class arrays "iou", "dice", "class_acc" and floats "miou", "mean_dice",
        "pixel_acc", "mean_class_acc". All values are fractions in 0-1.
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    gt_count = cm.sum(axis=1)
    pred_count = cm.sum(axis=0)
    fp = pred_count - tp
    fn = gt_count - tp

    iou = _safe_divide(tp, tp + fp + fn)
    dice = _safe_divide(2 * tp, 2 * tp + fp + fn)
    class_acc = _safe_divide(tp, gt_count)  # NaN for classes absent from the ground truth
    total = cm.sum()
    return {
        "iou": iou,
        "miou": _nanmean(iou),
        "dice": dice,
        "mean_dice": _nanmean(dice),
        "pixel_acc": float(tp.sum() / total) if total > 0 else float("nan"),
        "class_acc": class_acc,
        "mean_class_acc": _nanmean(class_acc),
    }


def image_scores(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Confusion matrix plus the scalar metrics for one image (for per_image.csv)."""
    cm = confusion_matrix(pred, gt)
    m = metrics_from_cm(cm)
    scalars = {k: float(v) for k, v in m.items() if np.ndim(v) == 0}
    return cm, scalars


def save_cm(path: str | Path, cm: np.ndarray) -> None:
    """Save one confusion matrix as a compressed .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, cm=cm.astype(np.int64))


def load_cm(path: str | Path) -> np.ndarray:
    """Load a confusion matrix saved with `save_cm`."""
    with np.load(path) as data:
        return data["cm"]
