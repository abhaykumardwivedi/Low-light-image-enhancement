"""Tests for lowlight.config."""

import pytest

from lowlight.config import (
    REPO_ROOT,
    ConfigError,
    deep_merge,
    load_config,
    load_darkness_levels,
    resolve_path,
)


def test_default_config_loads():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["device"] == "auto"
    assert cfg["inference_short_side"] == 512
    assert cfg["enhance_defaults"]["clahe"] == {"clip_limit": 2.0, "tile_grid": 8}


def test_experiment_file_and_overrides_are_merged(tmp_path):
    exp = tmp_path / "exp.yaml"
    exp.write_text("inference_short_side: 1024\nenhance_defaults:\n  gamma: {gamma: 0.4}\n")
    cfg = load_config(exp, overrides={"limit": 20})
    assert cfg["inference_short_side"] == 1024
    assert cfg["enhance_defaults"]["gamma"] == {"gamma": 0.4}
    # Settings the experiment did not mention keep their default values.
    assert cfg["enhance_defaults"]["clahe"]["tile_grid"] == 8
    assert cfg["limit"] == 20


def test_deep_merge_does_not_modify_inputs():
    base = {"a": {"b": 1, "c": 2}}
    merged = deep_merge(base, {"a": {"b": 9}})
    assert merged == {"a": {"b": 9, "c": 2}}
    assert base == {"a": {"b": 1, "c": 2}}


@pytest.mark.parametrize(
    "override",
    [
        {"device": "gpu"},
        {"seed": -1},
        {"seed": True},
        {"inference_short_side": 0},
        {"sliding_window": "no"},
        {"limit": 0},
        {"paths": {"results_root": ""}},
    ],
)
def test_invalid_values_raise(override):
    with pytest.raises(ConfigError):
        load_config(overrides=override)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_resolve_path():
    assert resolve_path("data") == (REPO_ROOT / "data").resolve()
    assert resolve_path(REPO_ROOT) == REPO_ROOT


def test_darkness_levels_load_and_get_darker():
    levels = load_darkness_levels()
    assert list(levels) == ["L0", "L1", "L2", "L3", "L4"]
    assert levels["L0"]["alpha"] == 1.0 and levels["L0"]["peak"] is None
    alphas = [levels[f"L{i}"]["alpha"] for i in range(5)]
    assert alphas == sorted(alphas, reverse=True)


def test_darkness_levels_reject_half_noise_settings(tmp_path):
    # L4 sets 'peak' but leaves 'sigma' empty, which is not allowed.
    rows = [f"  L{i}: {{alpha: 1.0, g: 1.0, peak: null, sigma: null}}" for i in range(4)]
    rows.append("  L4: {alpha: 0.2, g: 3.0, peak: 50, sigma: null}")
    bad = tmp_path / "dark.yaml"
    bad.write_text("levels:\n" + "\n".join(rows) + "\n")
    with pytest.raises(ConfigError, match="both"):
        load_darkness_levels(bad)
