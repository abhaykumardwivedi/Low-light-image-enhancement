"""Tests for lowlight.enhance (PID Section 11.3)."""

import math

import numpy as np
import pytest

from lowlight.enhance import (
    adaptive_gamma_value,
    available_methods,
    clahe,
    enhance,
    gamma,
    gamma_lut,
    mean_lightness,
    register,
)

METHODS = ["none", "gamma", "he", "clahe", "gamma_clahe", "adaptive_gamma"]


def test_all_pid_methods_registered():
    assert set(METHODS) <= set(available_methods())


@pytest.mark.parametrize("method", METHODS)
def test_shape_dtype_and_determinism(scene, method):
    out1 = enhance(scene, method)
    out2 = enhance(scene, method)
    assert out1.shape == scene.shape and out1.dtype == np.uint8
    np.testing.assert_array_equal(out1, out2)  # same input -> same output, every time
    assert out1 is not scene  # never returns the caller's array itself


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("shape", [(1, 1, 3), (3, 5, 3), (7, 13, 3), (33, 17, 3)])
def test_odd_and_tiny_sizes(method, shape):
    img = np.random.default_rng(1).integers(0, 256, size=shape, dtype=np.uint8)
    assert enhance(img, method).shape == shape


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("value", [0, 255, (90, 140, 30)])
def test_flat_images_do_not_crash(method, value):
    img = np.zeros((16, 24, 3), np.uint8)
    img[:] = value  # all black, all white, or one single colour
    out = enhance(img, method)
    assert out.shape == img.shape
    if method in ("he", "clahe"):
        np.testing.assert_array_equal(out, img)  # nothing to equalize: unchanged


def test_gamma_one_is_identity(scene):
    np.testing.assert_array_equal(enhance(scene, "gamma", gamma=1.0), scene)


def test_gamma_lut_values():
    lut = gamma_lut(0.5)
    assert lut[0] == 0 and lut[255] == 255
    assert lut[25] == 80 and lut[200] == 226  # the examples in the docstring
    assert np.all(np.diff(lut.astype(int)) >= 0)  # never reverses the brightness order


def test_gamma_brightens_dark_images(scene):
    dark = (scene // 4).astype(np.uint8)
    assert enhance(dark, "gamma", gamma=0.5).mean() > dark.mean()


def test_grayscale_and_rgba_inputs(scene):
    gray = scene[..., 0]
    assert enhance(gray, "clahe").shape == (64, 96, 3)
    rgba = np.concatenate([scene, np.full((64, 96, 1), 255, np.uint8)], axis=-1)
    np.testing.assert_array_equal(enhance(rgba, "none"), scene)


def test_bad_inputs_are_refused(scene):
    with pytest.raises(TypeError):
        enhance(scene.astype(np.float32), "gamma")
    with pytest.raises(KeyError):
        enhance(scene, "magic")
    with pytest.raises(ValueError):
        enhance(scene, "gamma", gamma=0)
    with pytest.raises(ValueError):
        enhance(scene, "clahe", clip_limit=-1)


def test_he_spreads_lightness(scene):
    low_contrast = (100 + scene // 8).astype(np.uint8)  # squeezed into 100-131
    out = enhance(low_contrast, "he")
    assert out.std() > 3 * low_contrast.std()


def test_l_channel_methods_keep_gray_gray():
    ramp = np.tile(np.linspace(20, 120, 50).astype(np.uint8), (20, 1))
    gray = np.stack([ramp] * 3, axis=-1)
    for method in ("he", "clahe", "gamma_clahe"):
        out = enhance(gray, method).astype(int)
        # Equalizing L only must not create colour: R, G and B stay (almost) equal.
        assert np.abs(out[..., 0] - out[..., 1]).max() <= 2
        assert np.abs(out[..., 1] - out[..., 2]).max() <= 2


def test_gamma_clahe_is_gamma_then_clahe(scene):
    expected = clahe(gamma(scene, gamma=0.6), clip_limit=3.0, tile_grid=4)
    out = enhance(scene, "gamma_clahe", gamma=0.6, clip_limit=3.0, tile_grid=4)
    np.testing.assert_array_equal(out, expected)


def test_adaptive_gamma_formula(scene):
    dark = (scene // 6).astype(np.uint8)
    m = mean_lightness(dark)
    expected = min(max(math.log(0.45) / math.log(m), 0.25), 1.0)
    assert adaptive_gamma_value(dark) == pytest.approx(expected)
    assert enhance(dark, "adaptive_gamma").mean() > dark.mean()


def test_adaptive_gamma_edge_cases(scene):
    bright = np.full((8, 8, 3), 230, np.uint8)
    assert adaptive_gamma_value(bright) == 1.0
    np.testing.assert_array_equal(enhance(bright, "adaptive_gamma"), bright)
    black = np.zeros((8, 8, 3), np.uint8)
    assert adaptive_gamma_value(black) == 0.25  # no log(0) crash; clipped to min_gamma


def test_duplicate_registration_refused():
    with pytest.raises(ValueError):
        register("gamma")(lambda img: img)
