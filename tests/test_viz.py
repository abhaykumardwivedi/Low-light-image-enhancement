"""Tests for lowlight.viz."""

import numpy as np
import pytest

from lowlight.labels import PALETTE
from lowlight.viz import colorize, comparison_grid, error_map, legend, overlay, plot_l_histograms


def test_colorize_uses_palette_and_black_for_ignore():
    mask = np.array([[0, 13], [255, 10]], np.uint8)
    rgb = colorize(mask)
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == tuple(PALETTE[0]) and tuple(rgb[0, 1]) == (0, 0, 142)
    assert tuple(rgb[1, 0]) == (0, 0, 0)


def test_overlay_blends_and_keeps_ignored_pixels():
    img = np.full((1, 2, 3), 100, np.uint8)
    mask = np.array([[13, 255]], np.uint8)  # car, ignore
    out = overlay(img, mask, alpha=0.5)
    assert tuple(out[0, 0]) == (50, 50, 121)  # halfway between gray 100 and car (0, 0, 142)
    assert tuple(out[0, 1]) == (100, 100, 100)


def test_error_map_colours():
    gt = np.array([[1, 2, 255]], np.uint8)
    pred = np.array([[1, 3, 4]], np.uint8)
    out = error_map(pred, gt)
    assert tuple(out[0, 0]) == (0, 200, 0)  # correct -> green
    assert tuple(out[0, 1]) == (220, 0, 0)  # wrong -> red
    assert tuple(out[0, 2]) == (0, 0, 0)  # ignored -> black


def test_comparison_grid_layout(scene):
    mask = np.zeros(scene.shape[:2], np.uint8)
    rows = [("none", scene, mask, mask, 0.5), ("gamma", scene, mask, None, None)]
    grid = comparison_grid(rows, tile_width=96)
    # 3 columns of width 96; each row = 28 px title + 64 px image (scene is 64 x 96)
    assert grid.shape == (2 * (28 + 64), 3 * 96, 3)


def test_legend_has_all_classes():
    assert legend(columns=4).shape == (5 * 24, 4 * 190, 3)


def test_histogram_chart(scene, tmp_path):
    pytest.importorskip("matplotlib")  # charts are drawn on Colab
    plot_l_histograms({"none": scene, "gamma": scene // 2}, tmp_path / "hist")
    assert (tmp_path / "hist.png").is_file() and (tmp_path / "hist.pdf").is_file()
