"""Classical low-light enhancement methods (PID Section 9.3).

IMAGE RULE (PID 11.1): every function here takes and returns an RGB uint8 NumPy array of shape
(H, W, 3). Images are converted from OpenCV's BGR order when they are loaded (datasets.py), so
nothing in this file ever sees BGR.

Every method is registered by name, so the rest of the project can say
`enhance(img, "clahe", clip_limit=2.0)` and new methods can be added with one decorator:

    @register("my_method")
    def my_method(img, strength=1.0): ...

HE, CLAHE and adaptive gamma work on the L (lightness) channel of the LAB colour space, so they
change brightness and contrast without shifting colours. Gamma works on all three RGB channels,
which keeps their ratios (and so the hue) roughly the same.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import cv2
import numpy as np

EnhanceFn = Callable[..., np.ndarray]
_REGISTRY: dict[str, EnhanceFn] = {}


def register(name: str) -> Callable[[EnhanceFn], EnhanceFn]:
    """Decorator that adds an enhancement function to the registry under `name`."""

    def decorator(fn: EnhanceFn) -> EnhanceFn:
        if name in _REGISTRY:
            raise ValueError(f"Enhancement method {name!r} is already registered")
        _REGISTRY[name] = fn
        return fn

    return decorator


def available_methods() -> list[str]:
    """Names of all registered methods, e.g. ['adaptive_gamma', 'clahe', ...]."""
    return sorted(_REGISTRY)


def ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Return `img` as a contiguous RGB uint8 (H, W, 3) array.

    Grayscale (H, W) or (H, W, 1) images are copied into three channels and RGBA images lose
    their transparency channel. Non-uint8 images are refused rather than silently rescaled,
    because a float image in 0-1 and one in 0-255 cannot be told apart reliably.
    """
    if img.dtype != np.uint8:
        raise TypeError(f"Expected a uint8 image, got {img.dtype}")
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    elif not (img.ndim == 3 and img.shape[2] == 3):
        raise ValueError(f"Expected an (H, W, 3) image, got shape {img.shape}")
    return np.ascontiguousarray(img)


def enhance(img: np.ndarray, method: str, **params: float) -> np.ndarray:
    """Apply a registered enhancement method.

    Args:
        img: RGB uint8 image (grayscale and RGBA are converted first).
        method: one of `available_methods()`.
        **params: method parameters, e.g. gamma=0.5 or clip_limit=2.0, tile_grid=8.

    Returns:
        A new RGB uint8 image with the same height and width.
    """
    if method not in _REGISTRY:
        raise KeyError(f"Unknown enhancement method {method!r}. Known: {available_methods()}")
    img = ensure_rgb_uint8(img)
    out = _REGISTRY[method](img, **params)
    # Every method must keep the image format, so later steps can rely on it.
    assert out.shape == img.shape and out.dtype == np.uint8, f"{method} broke the image format"
    return out


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def gamma_lut(gamma: float) -> np.ndarray:
    """256-entry lookup table: lut[v] = round(255 * (v / 255) ** gamma)."""
    if not gamma > 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")
    x = np.arange(256, dtype=np.float64) / 255.0
    return np.clip(np.rint(255.0 * x**gamma), 0, 255).astype(np.uint8)


def apply_gamma(img: np.ndarray, g: float) -> np.ndarray:
    """Apply gamma `g` to all channels through the lookup table (cv2.LUT is very fast)."""
    return cv2.LUT(img, gamma_lut(g))


def _apply_to_lightness(img: np.ndarray, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Run `fn` on the L channel of LAB and convert back to RGB.

    If the image has a single brightness value everywhere (all black, all white, one colour),
    there is no contrast to stretch, so the image is returned unchanged. This also avoids
    OpenCV turning a flat image into an arbitrary new value.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    lightness = lab[..., 0]
    if lightness.min() == lightness.max():
        return img.copy()
    lab[..., 0] = fn(np.ascontiguousarray(lightness))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def mean_lightness(img: np.ndarray) -> float:
    """Mean of the LAB L channel scaled to 0-1 (OpenCV stores L as 0-255 in 8-bit images)."""
    lightness = cv2.cvtColor(ensure_rgb_uint8(img), cv2.COLOR_RGB2LAB)[..., 0]
    return float(lightness.mean()) / 255.0


def adaptive_gamma_value(
    img: np.ndarray,
    target_mean: float = 0.45,
    min_gamma: float = 0.25,
    max_gamma: float = 1.0,
) -> float:
    """The gamma that would move the image's mean lightness m to `target_mean`.

    Solves m ** gamma = target_mean, i.e. gamma = log(target_mean) / log(m), then clips it to
    [min_gamma, max_gamma]. Images already at least as bright as the target get gamma = 1.0
    (no change). m is clamped to at least 1/255 so an all-black image does not give log(0).
    """
    if not 0 < target_mean < 1:
        raise ValueError(f"target_mean must be between 0 and 1, got {target_mean}")
    m = mean_lightness(img)
    if m >= target_mean:
        return 1.0
    m = min(max(m, 1.0 / 255.0), 1.0 - 1e-6)  # keep log(m) finite and non-zero
    return float(np.clip(math.log(target_mean) / math.log(m), min_gamma, max_gamma))


# --------------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------------


@register("none")
def none(img: np.ndarray) -> np.ndarray:
    """No enhancement: returns a copy of the input. This is the baseline every method is
    compared against.

    Analogy: the "control group" in an experiment.
    """
    return img.copy()


@register("gamma")
def gamma(img: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """Gamma correction: out = 255 * (in / 255) ** gamma, on all three channels.

    gamma < 1 lifts dark values a lot and bright values only a little (0.5 maps 25 -> 80 but
    200 -> 226); gamma = 1 changes nothing. Implemented with a 256-entry lookup table, because a
    uint8 pixel has only 256 possible values, so we compute each answer once.

    Analogy: turning up the "shadows" slider in a photo app.
    """
    return apply_gamma(img, gamma)


@register("he")
def he(img: np.ndarray) -> np.ndarray:
    """Global histogram equalization of the L channel (LAB colour space).

    Each lightness value is replaced by its cumulative share of pixels (the CDF), so the output
    lightness values spread over the full 0-255 range. Done on L only, because equalizing R, G
    and B separately changes the colours.

    Analogy: re-spacing runners on a track so they use the whole track, not one corner.
    """
    return _apply_to_lightness(img, cv2.equalizeHist)


@register("clahe")
def clahe(img: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """CLAHE: histogram equalization done separately on a tile_grid x tile_grid grid of tiles,
    with each tile's histogram clipped at `clip_limit` (relative to a flat histogram) and the
    clipped excess spread evenly. Neighbouring tiles are blended with bilinear interpolation.

    The clip limit stops flat, noisy regions (like a night sky) from being stretched into
    strong noise; the tiles let dark and bright areas of the same image each get their own
    contrast boost.

    Analogy: HE with a separate, gentle volume control for each part of the picture.
    """
    if not clip_limit > 0:
        raise ValueError(f"clip_limit must be > 0, got {clip_limit}")
    if int(tile_grid) < 1:
        raise ValueError(f"tile_grid must be >= 1, got {tile_grid}")
    tiles = (int(tile_grid), int(tile_grid))
    op = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tiles)
    return _apply_to_lightness(img, op.apply)


@register("gamma_clahe")
def gamma_clahe(
    img: np.ndarray, gamma: float = 0.5, clip_limit: float = 2.0, tile_grid: int = 8
) -> np.ndarray:
    """Gamma correction first, then CLAHE.

    The order matters: gamma first lifts the dark values out of the bottom few histogram bins,
    so CLAHE then has real detail to work with instead of a few crowded dark levels.

    Analogy: first switch on the room light, then adjust the contrast.
    """
    return clahe(apply_gamma(img, gamma), clip_limit=clip_limit, tile_grid=tile_grid)


@register("adaptive_gamma")
def adaptive_gamma(
    img: np.ndarray,
    target_mean: float = 0.45,
    min_gamma: float = 0.25,
    max_gamma: float = 1.0,
) -> np.ndarray:
    """Gamma correction whose gamma is chosen from the image itself.

    gamma = log(target_mean) / log(m), where m is the image's mean lightness (0-1), clipped to
    [min_gamma, max_gamma]; bright images (m >= target_mean) are left unchanged. So a very dark
    image gets a strong boost and a dusk image only a light one.

    Analogy: auto-brightness on a phone screen.
    """
    g = adaptive_gamma_value(img, target_mean, min_gamma, max_gamma)
    return img.copy() if g == 1.0 else apply_gamma(img, g)
