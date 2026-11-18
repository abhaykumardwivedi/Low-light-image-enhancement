"""Cheap statistics of an input image, used by the adaptive selector (PID Section 9.8) and by the
darkness calibration.

All lightness values are on a 0-1 scale (decision D3): OpenCV's 8-bit LAB stores L as 0-255, so
we divide by 255.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from lowlight.enhance import ensure_rgb_uint8

# Pixels at least this bright are treated as light sources (street lamps, headlights).
LIGHT_SOURCE_THRESHOLD = 0.98

FEATURE_NAMES = (
    "mean_l",
    "median_l",
    "std_l",
    "frac_dark",
    "frac_bright",
    "rms_contrast",
    "entropy",
    "noise_sigma",
    "light_source_ratio",
)


def lightness(img: np.ndarray) -> np.ndarray:
    """The LAB L channel as float32 in 0-1."""
    lab = cv2.cvtColor(ensure_rgb_uint8(img), cv2.COLOR_RGB2LAB)
    return lab[..., 0].astype(np.float32) / 255.0


def l_histogram(img: np.ndarray, bins: int = 256) -> np.ndarray:
    """Normalised histogram of the L channel (sums to 1), for before/after plots."""
    hist, _ = np.histogram(lightness(img), bins=bins, range=(0.0, 1.0))
    return hist / max(hist.sum(), 1)


def entropy(img: np.ndarray) -> float:
    """Shannon entropy of the L channel in bits (0 = one flat value, 8 = all 256 values equally).

    Low entropy means few distinct brightness levels, i.e. little visible detail.
    """
    p = l_histogram(img)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def estimate_noise(img: np.ndarray) -> float:
    """Estimate the standard deviation of Gaussian noise (0-1 scale) with Immerkær's method.

    The image is filtered with a small kernel that cancels smooth brightness changes and edges
    of first order, so mostly noise is left. Its mean absolute response is converted to a sigma:
        sigma = sqrt(pi / 2) * sum(|I * M|) / (6 * (W - 2) * (H - 2))
    Reference: J. Immerkær, "Fast Noise Variance Estimation", CVIU 64(2), 1996.
    We use this instead of scikit-image's estimate_sigma because it needs no extra package and
    is fast enough to run on every image (decision D31).
    """
    gray = cv2.cvtColor(ensure_rgb_uint8(img), cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float64)
    response = cv2.filter2D(gray, -1, kernel)[1:-1, 1:-1]  # drop the border rows/columns
    return float(math.sqrt(math.pi / 2) * np.abs(response).sum() / (6 * (w - 2) * (h - 2)))


def compute_features(img: np.ndarray) -> dict[str, float]:
    """Compute every feature in FEATURE_NAMES for one RGB uint8 image."""
    img = ensure_rgb_uint8(img)
    lum = lightness(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return {
        "mean_l": float(lum.mean()),
        "median_l": float(np.median(lum)),
        "std_l": float(lum.std()),
        "frac_dark": float((lum < 0.1).mean()),
        "frac_bright": float((lum > 0.9).mean()),
        "rms_contrast": float(gray.std()),  # RMS contrast = standard deviation of intensity
        "entropy": entropy(img),
        "noise_sigma": estimate_noise(img),
        "light_source_ratio": float((lum >= LIGHT_SOURCE_THRESHOLD).mean()),
    }
