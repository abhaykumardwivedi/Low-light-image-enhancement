"""Tests for lowlight.features."""

import numpy as np
import pytest

from lowlight.darken import darken
from lowlight.features import FEATURE_NAMES, compute_features, entropy, estimate_noise


def test_all_features_present(scene):
    feats = compute_features(scene)
    assert tuple(feats) == FEATURE_NAMES
    assert all(isinstance(v, float) for v in feats.values())


def test_black_and_white_extremes():
    black = compute_features(np.zeros((10, 10, 3), np.uint8))
    assert black["mean_l"] == 0 and black["frac_dark"] == 1 and black["entropy"] == 0
    assert black["noise_sigma"] == 0
    white = compute_features(np.full((10, 10, 3), 255, np.uint8))
    assert white["mean_l"] == pytest.approx(1.0)
    assert white["frac_bright"] == 1 and white["light_source_ratio"] == 1


def test_entropy_of_two_equal_levels_is_one_bit():
    img = np.zeros((10, 10, 3), np.uint8)
    img[:5] = 255
    assert entropy(img) == pytest.approx(1.0)


@pytest.mark.parametrize("sigma", [0.02, 0.05])
def test_noise_estimate_close_to_truth(sigma):
    rng = np.random.default_rng(3)
    noisy = 0.5 + rng.normal(0, sigma, (200, 200))
    img = np.repeat((np.clip(noisy, 0, 1) * 255).round().astype(np.uint8)[..., None], 3, axis=2)
    assert estimate_noise(img) == pytest.approx(sigma, rel=0.15)


def test_darker_level_gives_darker_features(scene):
    f1 = compute_features(darken(scene, "L1", seed=0, image_id="s"))
    f4 = compute_features(darken(scene, "L4", seed=0, image_id="s"))
    assert f4["mean_l"] < f1["mean_l"] and f4["frac_dark"] > f1["frac_dark"]
