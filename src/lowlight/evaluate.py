"""Evaluate one model on one dataset under one condition (darkness level + enhancement method).

This is the inner loop of every experiment: for each image, optionally darken it, enhance it,
segment it, and compare the prediction with the ground truth. The M4 experiment runner calls this
for every condition and adds saving, resuming and logging around it.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lowlight.darken import darken
from lowlight.datasets import SegDataset
from lowlight.enhance import enhance
from lowlight.metrics import image_scores, metrics_from_cm
from lowlight.models import Segmenter


@dataclass
class EvalResult:
    """Everything measured for one condition."""

    image_ids: list[str]
    cms: np.ndarray  # (N, C, C) per-image confusion matrices
    per_image: list[dict[str, Any]]  # one row per image: id, miou, mean_dice, pixel_acc, times
    summary: dict[str, Any] = field(default_factory=dict)  # dataset-level metrics from summed cms


def evaluate(
    segmenter: Segmenter,
    dataset: SegDataset,
    method: str = "none",
    params: dict[str, Any] | None = None,
    level: str | None = None,
    noise: bool = True,
    seed: int = 42,
    on_image: Callable[[str, np.ndarray, np.ndarray, np.ndarray], None] | None = None,
) -> EvalResult:
    """Run darken (optional) -> enhance -> segment -> score over every image.

    Args:
        level: darkness level ("L0".."L4") for the controlled experiment, or None for real images.
        on_image: optional callback(image_id, enhanced image, prediction, ground truth), e.g. to
            save example pictures.

    Timing: enhancement and inference are timed separately in milliseconds with a monotonic
    clock. (The careful speed study with warm-up runs is experiment E5.)
    """
    params = params or {}
    ids, cms, rows = [], [], []
    for img, gt, sample in dataset:
        if gt is None:
            raise ValueError(f"{sample.image_id} has no label; evaluation needs ground truth")
        if level is not None:
            img = darken(img, level, seed, noise=noise, image_id=sample.image_id)

        t0 = time.perf_counter()
        enhanced = enhance(img, method, **params)
        t1 = time.perf_counter()
        pred = segmenter.predict(enhanced)
        t2 = time.perf_counter()

        cm, scores = image_scores(pred, gt)
        ids.append(sample.image_id)
        cms.append(cm)
        rows.append(
            {
                "image_id": sample.image_id,
                **scores,
                "enhance_ms": 1000 * (t1 - t0),
                "infer_ms": 1000 * (t2 - t1),
            }
        )
        if on_image is not None:
            on_image(sample.image_id, enhanced, pred, gt)

    stacked = np.stack(cms) if cms else np.zeros((0, 19, 19), np.int64)
    total = metrics_from_cm(stacked.sum(axis=0))
    summary = {
        "n_images": len(ids),
        "miou": total["miou"],
        "mean_dice": total["mean_dice"],
        "pixel_acc": total["pixel_acc"],
        "mean_class_acc": total["mean_class_acc"],
        "per_class_iou": total["iou"].tolist(),
        "median_enhance_ms": float(np.median([r["enhance_ms"] for r in rows])) if rows else None,
        "median_infer_ms": float(np.median([r["infer_ms"] for r in rows])) if rows else None,
    }
    return EvalResult(ids, stacked, rows, summary)
