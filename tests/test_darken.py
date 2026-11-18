"""Tests for lowlight.darken (PID Section 11.3)."""

import numpy as np
import pytest

from lowlight.darken import darken, darken_with_params, image_seed


def test_l0_returns_input_exactly(scene):
    out = darken(scene, "L0", seed=42, image_id="a")
    np.testing.assert_array_equal(out, scene)
    assert out is not scene


def test_brightness_strictly_decreases_l1_to_l4(scene):
    means = [darken(scene, f"L{i}", seed=42, image_id="a").mean() for i in range(5)]
    assert all(a > b for a, b in zip(means, means[1:], strict=False)), means


def test_same_seed_same_output_different_id_different_noise(scene):
    a1 = darken(scene, "L3", seed=42, image_id="frankfurt_000000_000294")
    a2 = darken(scene, "L3", seed=42, image_id="frankfurt_000000_000294")
    b = darken(scene, "L3", seed=42, image_id="lindau_000000_000019")
    c = darken(scene, "L3", seed=7, image_id="frankfurt_000000_000294")
    np.testing.assert_array_equal(a1, a2)
    assert not np.array_equal(a1, b) and not np.array_equal(a1, c)


def test_image_seed_is_stable_across_runs():
    # CRC32("abc") = 0x352441C2. A fixed number proves the seed does not depend on Python's
    # randomised hash(), so the noise is the same on the laptop, on Colab and next year.
    assert image_seed(0, "abc") == 0x352441C2
    assert image_seed(10, "abc") == 0x352441C2 + 10


def test_noise_free_ablation_matches_formula(scene):
    out = darken(scene, "L3", seed=1, image_id="x", noise=False)
    x = scene / 255.0
    expected = np.rint(np.clip(0.35 * x**2.5, 0, 1) * 255).astype(np.uint8)
    np.testing.assert_array_equal(out, expected)
    noisy = darken(scene, "L3", seed=1, image_id="x", noise=True)
    assert not np.array_equal(out, noisy)
    assert abs(float(out.mean()) - float(noisy.mean())) < 2  # noise adds no brightness bias


def test_output_format_and_errors(scene):
    out = darken(scene, "L4", seed=0, image_id="x")
    assert out.dtype == np.uint8 and out.shape == scene.shape
    with pytest.raises(KeyError):
        darken(scene, "L9", seed=0)
    with pytest.raises(TypeError):
        darken(scene.astype(np.float32), "L1", seed=0)
    with pytest.raises(ValueError):
        darken_with_params(scene, 0.5, 2.0, peak=100, sigma=0.01, rng=None)
