"""Tests for lowlight.stats (PID Section 11.3: bootstrap CI contains the true difference)."""

import numpy as np
import pytest

from lowlight.stats import compare_to_baseline, dataset_miou, paired_bootstrap, wilcoxon_p


def _population(n, rng, boost):
    """Per-image 2-class confusion matrices; `boost` moves pixels from wrong to right."""
    cms_base, cms_method = [], []
    for _ in range(n):
        size = rng.integers(500, 1500)
        right = rng.integers(int(0.5 * size), int(0.8 * size))
        wrong = size - right
        better = min(wrong, int(rng.normal(boost * size, 0.03 * size)) if boost else 0)
        better = max(better, -right)
        base = np.array([[right // 2, wrong // 2], [wrong - wrong // 2, right - right // 2]])
        moved = np.array(
            [[better // 2, -(better // 2)], [-(better - better // 2), better - better // 2]]
        )
        cms_base.append(base)
        cms_method.append(base + moved)
    return np.array(cms_base), np.array(cms_method)


def test_ci_contains_true_difference():
    rng = np.random.default_rng(1)
    pop_base, pop_method = _population(20000, rng, boost=0.03)
    true_diff = dataset_miou(pop_method) - dataset_miou(pop_base)  # "the whole world"
    idx = rng.choice(20000, 150, replace=False)  # our test set is a sample of it
    observed, low, high = paired_bootstrap(pop_method[idx], pop_base[idx], n_boot=1000, seed=42)
    assert low <= true_diff <= high
    assert low <= observed <= high
    assert low > 0  # a consistent improvement is detected as significant


def test_no_difference_gives_zero_interval():
    rng = np.random.default_rng(2)
    base, _ = _population(50, rng, boost=0)
    assert paired_bootstrap(base, base) == (0.0, 0.0, 0.0)


def test_bootstrap_is_reproducible_and_antisymmetric():
    rng = np.random.default_rng(3)
    base, method = _population(60, rng, boost=0.02)
    assert paired_bootstrap(method, base, seed=7) == paired_bootstrap(method, base, seed=7)
    d1, lo1, hi1 = paired_bootstrap(method, base, seed=7)
    d2, lo2, hi2 = paired_bootstrap(base, method, seed=7)
    assert d1 == pytest.approx(-d2) and lo1 == pytest.approx(-hi2) and hi1 == pytest.approx(-lo2)


def test_bootstrap_needs_same_images():
    with pytest.raises(ValueError):
        paired_bootstrap(np.zeros((3, 2, 2)), np.zeros((4, 2, 2)))


def test_comparison_significance_flag():
    rng = np.random.default_rng(4)
    base, method = _population(80, rng, boost=0.03)
    per_base = np.array([dataset_miou(c[None]) for c in base])
    per_method = np.array([dataset_miou(c[None]) for c in method])
    result = compare_to_baseline(method, base, per_method, per_base, n_boot=300)
    assert result.significant and result.dataset_diff > 0 and result.n_images == 80
    assert result.mean_image_diff > 0


def test_wilcoxon():
    pytest.importorskip("scipy")  # installed on Colab; skipped on the laptop
    rng = np.random.default_rng(5)
    base = rng.uniform(0.3, 0.6, 40)
    assert wilcoxon_p(base + 0.05 + rng.normal(0, 0.01, 40), base) < 0.001
    assert wilcoxon_p(base, base) == 1.0
