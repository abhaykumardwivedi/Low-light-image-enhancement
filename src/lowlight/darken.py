"""Synthetic low-light generation (PID Section 9.2).

Turns a well-lit image into a controlled "night" version, so the same scene with the same labels
can be tested at known darkness levels L0 (unchanged) to L4 (very dark).

Model, with pixel values scaled to 0-1:
    1. d = alpha * x ** g                        (less light, and lower contrast in the shadows)
    2. d = Poisson(d * peak) / peak              (shot noise: light arrives as random photons)
    3. d = d + Normal(0, sigma)                  (read noise from the camera electronics)
    4. clip to [0, 1] and convert back to uint8

IMAGE RULE: input and output are RGB uint8 (H, W, 3) arrays.
"""

from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from lowlight.config import load_darkness_levels


def image_seed(seed: int, image_id: str) -> int:
    """A reproducible random seed for one image: seed + CRC32(image_id).

    Python's built-in hash() of a string changes every time Python starts (hash randomisation),
    so it would give different noise on every run. CRC32 always gives the same number for the
    same text (decision D2 in docs/PROGRESS.md).
    """
    return (seed + zlib.crc32(image_id.encode("utf-8"))) % 2**32


def darken_with_params(
    img: np.ndarray,
    alpha: float,
    g: float,
    peak: float | None,
    sigma: float | None,
    rng: np.random.Generator | None = None,
    noise: bool = True,
) -> np.ndarray:
    """Darken an RGB uint8 image with explicit parameters (see the module docstring)."""
    if img.dtype != np.uint8:
        raise TypeError(f"Expected a uint8 image, got {img.dtype}")
    if alpha == 1.0 and g == 1.0 and (not noise or peak is None):
        return img.copy()  # L0 or a no-op setting: return the input exactly

    x = img.astype(np.float64) / 255.0
    d = alpha * x**g
    if noise and peak is not None:
        if rng is None:
            raise ValueError("A random generator is needed when noise is on")
        d = rng.poisson(d * peak) / peak  # brighter pixels get more photons, so more noise
        d = d + rng.normal(0.0, sigma or 0.0, size=d.shape)
    return np.rint(np.clip(d, 0.0, 1.0) * 255.0).astype(np.uint8)


def darken(
    img: np.ndarray,
    level: str,
    seed: int,
    noise: bool = True,
    image_id: str = "",
    levels: dict[str, dict[str, Any]] | None = None,
) -> np.ndarray:
    """Darken an image to one of the levels L0-L4 from configs/darkness_levels.yaml.

    Args:
        img: RGB uint8 image.
        level: "L0" ... "L4".
        seed: the experiment seed (from the config).
        noise: False runs the "brightness only" ablation (no shot or read noise).
        image_id: the image's id; together with `seed` it fixes the noise for this image, so the
            same image always gets exactly the same noise.
        levels: level parameters (loaded from the YAML file if not given).
    """
    levels = levels if levels is not None else load_darkness_levels()
    if level not in levels:
        raise KeyError(f"Unknown darkness level {level!r}. Known: {sorted(levels)}")
    p = levels[level]
    rng = np.random.default_rng(image_seed(seed, image_id))
    return darken_with_params(img, p["alpha"], p["g"], p.get("peak"), p.get("sigma"), rng, noise)
