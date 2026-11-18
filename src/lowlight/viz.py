"""Pictures and charts (PID Section 9.9).

Image helpers (colourised masks, overlays, error maps, comparison grids, legend) use only NumPy and
OpenCV, so they run anywhere. Charts use matplotlib, which is imported only inside the chart
functions (it is preinstalled on Colab).

IMAGE RULE: images are RGB uint8 (H, W, 3); masks are uint8 (H, W) trainIds with 255 = ignore.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from lowlight.labels import CLASSES, IGNORE_COLOR, IGNORE_INDEX, PALETTE

# One fixed colour per method in every chart (Okabe-Ito palette, readable with colour blindness).
METHOD_COLORS = {
    "none": "#000000",
    "gamma": "#E69F00",
    "he": "#56B4E9",
    "clahe": "#009E73",
    "gamma_clahe": "#D55E00",
    "adaptive_gamma": "#CC79A7",
    "agcwd": "#0072B2",
    "zero_dce_pp": "#F0E442",
}

_MASK_LUT = np.zeros((256, 3), dtype=np.uint8)
_MASK_LUT[: len(PALETTE)] = PALETTE
_MASK_LUT[IGNORE_INDEX] = IGNORE_COLOR


def colorize(mask: np.ndarray) -> np.ndarray:
    """Turn a trainId mask into an RGB picture with the Cityscapes colours (ignore = black)."""
    return _MASK_LUT[mask]


def overlay(img: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Blend the coloured mask over the image; ignored pixels show the plain image."""
    color = colorize(mask).astype(np.float32)
    out = img.astype(np.float32)
    keep = mask != IGNORE_INDEX
    out[keep] = (1 - alpha) * out[keep] + alpha * color[keep]
    return np.rint(out).astype(np.uint8)


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Green = correct, red = wrong, black = ignored in the ground truth."""
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    valid = gt != IGNORE_INDEX
    out[valid & (pred == gt)] = (0, 200, 0)
    out[valid & (pred != gt)] = (220, 0, 0)
    return out


def _label_bar(text: str, width: int, height: int = 28) -> np.ndarray:
    bar = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(
        bar, text, (6, height - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA
    )
    return bar


def _fit(img: np.ndarray, width: int) -> np.ndarray:
    """Resize to `width` keeping the aspect ratio (nearest neighbour keeps mask colours exact)."""
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_NEAREST)


def comparison_grid(
    rows: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, float | None]],
    tile_width: int = 320,
) -> np.ndarray:
    """One row per method: [enhanced image | predicted mask | error map].

    Args:
        rows: (method name, enhanced image, predicted mask, ground truth or None, mIoU or None).
            Without ground truth, the third column is left blank.
    """
    out_rows = []
    for name, img, pred, gt, miou in rows:
        tiles = [_fit(img, tile_width), _fit(colorize(pred), tile_width)]
        if gt is not None:
            tiles.append(_fit(error_map(pred, gt), tile_width))
        else:
            tiles.append(np.full_like(tiles[0], 255))
        row = np.concatenate(tiles, axis=1)
        title = name if miou is None or np.isnan(miou) else f"{name}   mIoU {100 * miou:.1f}%"
        out_rows.append(np.concatenate([_label_bar(title, row.shape[1]), row], axis=0))
    return np.concatenate(out_rows, axis=0)


def legend(columns: int = 4, cell: tuple[int, int] = (190, 24)) -> np.ndarray:
    """A picture of the 19 class colours with their names."""
    cw, ch = cell
    n_rows = -(-len(CLASSES) // columns)  # ceiling division
    img = np.full((n_rows * ch, columns * cw, 3), 255, dtype=np.uint8)
    for i, name in enumerate(CLASSES):
        r, c = divmod(i, columns)
        x, y = c * cw, r * ch
        img[y + 4 : y + ch - 4, x + 4 : x + 24] = PALETTE[i]
        cv2.putText(img, name, (x + 30, y + ch - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    return img


# --------------------------------------------------------------------------------------------
# Charts (matplotlib)
# --------------------------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")  # draw to files, no screen needed
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, path_without_ext: str | Path) -> None:
    """Save a chart as PNG (200 dpi) and PDF, as the PID asks for the report."""
    path = Path(path_without_ext)
    path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")


def plot_l_histograms(images: dict[str, np.ndarray], path_without_ext: str | Path) -> None:
    """L-channel histogram of each version of one image (e.g. none, gamma, clahe ...)."""
    from lowlight.features import l_histogram

    plt = _plt()
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.linspace(0, 1, 256)
    for name, img in images.items():
        ax.plot(x, l_histogram(img), label=name, color=METHOD_COLORS.get(name), linewidth=1.5)
    ax.set_xlabel("lightness L (0 = black, 1 = white)")
    ax.set_ylabel("share of pixels")
    ax.set_title("Lightness histogram before and after enhancement")
    ax.legend()
    save_figure(fig, path_without_ext)
    plt.close(fig)
