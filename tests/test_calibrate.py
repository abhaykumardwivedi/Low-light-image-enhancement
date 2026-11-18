"""Tests for lowlight.calibrate (no real data needed)."""

from lowlight.calibrate import calibrate_levels, fit_alpha, image_stats, level_stats, levels_to_yaml
from lowlight.config import load_darkness_levels


def test_fit_alpha_hits_target(scene):
    day = [("a", scene)]
    params = {"alpha": 0.5, "g": 2.0, "peak": None, "sigma": None}
    alpha = fit_alpha(day, params, target_mean=0.2, seed=0)
    got = level_stats(day, {"X": {**params, "alpha": alpha}}, seed=0)["X"].mean_l
    assert abs(got - 0.2) < 0.01


def test_calibration_brackets_night_mean(scene):
    day = [("a", scene), ("b", scene[::-1])]
    levels = load_darkness_levels()
    before = level_stats(day, levels, seed=0)
    # Choose a "real night" mean that the starting values do NOT bracket (darker than L4).
    night_mean = before["L4"].mean_l * 0.6
    new, report = calibrate_levels(day, night_mean, levels, seed=0)
    assert not report["already_bracketed"] and report["bracketed_after"]
    after = report["after"]
    assert after["L3"]["mean_l"] >= night_mean >= after["L4"]["mean_l"]
    # Brightness must still go down from L0 to L4.
    means = [after[f"L{i}"]["mean_l"] for i in range(5)]
    assert means == sorted(means, reverse=True)
    assert new["L3"]["g"] == levels["L3"]["g"]  # only alpha is changed


def test_already_bracketed_changes_nothing(scene):
    day = [("a", scene)]
    levels = load_darkness_levels()
    s = level_stats(day, levels, seed=0)
    night_mean = (s["L3"].mean_l + s["L4"].mean_l) / 2
    new, report = calibrate_levels(day, night_mean, levels, seed=0)
    assert report["already_bracketed"] and new == levels


def test_yaml_round_trip(tmp_path):
    levels = load_darkness_levels()
    text = levels_to_yaml(levels, "test note")
    assert "CALIBRATED. test note" in text
    path = tmp_path / "d.yaml"
    path.write_text(text)
    assert load_darkness_levels(path) == levels


def test_image_stats(scene):
    s = image_stats([scene])
    assert 0 < s.mean_l < 1 and s.rms_contrast > 0
