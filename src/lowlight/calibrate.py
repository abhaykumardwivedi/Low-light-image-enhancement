"""Calibrate the synthetic darkness levels against real night images (PID Section 9.2).

Goal: the darkened daytime images at L3 and L4 should bracket the real night brightness, i.e.
mean lightness(L3) >= real night mean >= mean lightness(L4). Then our "night" levels are
realistic, and L3/L4 cover the range of real night scenes.

Only image statistics (brightness) are used, never labels or model accuracy, so using the
evaluation images' brightness here does not leak any result into the tuning (decision D32).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from lowlight.darken import darken
from lowlight.features import lightness


@dataclass
class LevelStats:
    """Mean lightness and RMS contrast (both 0-1), averaged over images."""

    mean_l: float
    rms_contrast: float


def image_stats(images: Iterable[np.ndarray]) -> LevelStats:
    """Average mean-lightness and contrast over a set of RGB uint8 images."""
    means, contrasts = [], []
    for img in images:
        lum = lightness(img)
        means.append(float(lum.mean()))
        contrasts.append(float(lum.std()))
    if not means:
        raise ValueError("No images given")
    return LevelStats(float(np.mean(means)), float(np.mean(contrasts)))


def level_stats(
    day_images: list[tuple[str, np.ndarray]],
    levels: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, LevelStats]:
    """Statistics of the day images after darkening to each level. Items are (image_id, image)."""
    return {
        name: image_stats(
            darken(img, name, seed, image_id=iid, levels=levels) for iid, img in day_images
        )
        for name in levels
    }


def _mean_at_alpha(day_images, params, alpha, seed) -> float:
    trial = {"X": {**params, "alpha": alpha}}
    return level_stats(day_images, trial, seed)["X"].mean_l


def fit_alpha(
    day_images: list[tuple[str, np.ndarray]],
    params: dict[str, Any],
    target_mean: float,
    seed: int,
    iterations: int = 20,
) -> float:
    """Find the alpha (with g and noise kept) that gives mean lightness `target_mean`.

    Mean lightness grows steadily with alpha, so a bisection search between 0.01 and 1.0 finds it.
    """
    lo, hi = 0.01, 1.0
    for _ in range(iterations):
        mid = (lo + hi) / 2
        if _mean_at_alpha(day_images, params, mid, seed) < target_mean:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def calibrate_levels(
    day_images: list[tuple[str, np.ndarray]],
    night_mean: float,
    levels: dict[str, dict[str, Any]],
    seed: int,
    margin: float = 0.15,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Adjust L3 and L4 so they bracket the real night mean lightness.

    If they already bracket it, nothing changes. Otherwise alpha of L3 is fitted so L3's mean is
    `margin` (15%) above the night mean, and alpha of L4 so L4's mean is 15% below it. g, peak and
    sigma are not changed. L1 and L2 are left alone unless they would become darker than L3.

    Returns:
        (new_levels, report) where report records the before/after numbers.
    """
    before = level_stats(day_images, levels, seed)
    new = {name: dict(p) for name, p in levels.items()}
    bracketed = before["L3"].mean_l >= night_mean >= before["L4"].mean_l
    if not bracketed:
        new["L3"]["alpha"] = fit_alpha(day_images, levels["L3"], night_mean * (1 + margin), seed)
        new["L4"]["alpha"] = fit_alpha(day_images, levels["L4"], night_mean * (1 - margin), seed)
        for upper in ("L2", "L1"):  # keep the levels in order: L1 >= L2 >= L3 in alpha
            below = "L3" if upper == "L2" else "L2"
            new[upper]["alpha"] = max(new[upper]["alpha"], new[below]["alpha"])
    after = level_stats(day_images, new, seed)
    report = {
        "night_mean_l": night_mean,
        "already_bracketed": bracketed,
        "before": {k: vars(v) for k, v in before.items()},
        "after": {k: vars(v) for k, v in after.items()},
        "bracketed_after": after["L3"].mean_l >= night_mean >= after["L4"].mean_l,
    }
    return new, report


def levels_to_yaml(levels: dict[str, dict[str, Any]], note: str) -> str:
    """Write the levels in the same layout as configs/darkness_levels.yaml, with a note."""
    meaning = {
        "L0": "original, no change",
        "L1": "dusk",
        "L2": "poorly lit street",
        "L3": "night with street lights",
        "L4": "very dark",
    }

    def fmt(v):
        return "null" if v is None else repr(v)

    lines = [
        "# Synthetic darkening levels (PID Section 9.2).",
        "# Formula (values in 0-1): d = alpha * x ** g, then optional Poisson noise with `peak`",
        "# photons and Gaussian read noise with std `sigma`.",
        "#",
        f"# STATUS: CALIBRATED. {note}",
        "",
        "levels:",
    ]
    for name, p in levels.items():
        body = ", ".join(f"{k}: {fmt(p.get(k))}" for k in ("alpha", "g", "peak", "sigma"))
        lines.append(f"  {name}: {{{body}}}  # {meaning.get(name, '')}")
    return "\n".join(lines) + "\n"
