"""Statistics for comparing an enhancement method against `none` (PID Section 9.6).

Two questions, two tests:
1. Paired bootstrap: "how sure are we about the change in dataset mIoU?" It gives a 95% confidence
   interval (CI) for mIoU(method) - mIoU(none). We call a change significant only when the CI does
   not contain zero.
2. Wilcoxon signed-rank test: "does the method tend to raise or lower each image's mIoU?" It looks
   at the per-image differences without assuming they follow a bell curve.
Both are PAIRED: the method and `none` are always compared on the same images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lowlight.metrics import metrics_from_cm


def dataset_miou(cms: np.ndarray, weights: np.ndarray | None = None) -> float:
    """mIoU of the summed confusion matrices (optionally each image counted `weights[i]` times)."""
    total = cms.sum(axis=0) if weights is None else np.tensordot(weights, cms, axes=1)
    return float(metrics_from_cm(total)["miou"])


def paired_bootstrap(
    cms_method: np.ndarray,
    cms_base: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float]:
    """Paired bootstrap CI for dataset mIoU(method) - mIoU(base).

    Each round draws N images WITH replacement (some twice, some not at all), using the SAME
    draw for both methods, and recomputes both dataset mIoUs from the summed matrices. The middle
    95% of the 1,000 differences is the confidence interval.

    Args:
        cms_method, cms_base: per-image confusion matrices, shape (N, C, C), same image order.

    Returns:
        (observed difference, CI low, CI high)
    """
    if cms_method.shape != cms_base.shape:
        raise ValueError("Both methods must be evaluated on the same images")
    n = cms_method.shape[0]
    rng = np.random.default_rng(seed)
    observed = dataset_miou(cms_method) - dataset_miou(cms_base)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        # How many times each image was drawn: summing weighted matrices is much faster than
        # building a new list of N matrices every round.
        weights = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float64)
        diffs[b] = dataset_miou(cms_method, weights) - dataset_miou(cms_base, weights)
    tail = (1 - ci) / 2 * 100
    low, high = np.percentile(diffs, [tail, 100 - tail])
    return float(observed), float(low), float(high)


def wilcoxon_p(per_image_method: np.ndarray, per_image_base: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value on per-image mIoU (pairs with a NaN are dropped).

    Returns 1.0 when every pair is identical (no evidence of any difference).
    """
    from scipy.stats import wilcoxon  # imported here so the rest of the module needs no scipy

    a, b = np.asarray(per_image_method, float), np.asarray(per_image_base, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    d = a[keep] - b[keep]
    if d.size == 0 or np.all(d == 0):
        return 1.0
    return float(wilcoxon(a[keep], b[keep]).pvalue)


@dataclass
class Comparison:
    """Result of comparing one method with `none`. All mIoU values are fractions (0-1)."""

    dataset_diff: float  # dataset mIoU(method) - mIoU(none)
    ci_low: float
    ci_high: float
    mean_image_diff: float  # mean over images of per-image mIoU difference
    wilcoxon_p: float
    n_images: int

    @property
    def significant(self) -> bool:
        """True only when the 95% CI excludes zero (PID 9.6)."""
        return self.ci_low > 0 or self.ci_high < 0


def compare_to_baseline(
    cms_method: np.ndarray,
    cms_base: np.ndarray,
    per_image_method: np.ndarray,
    per_image_base: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
) -> Comparison:
    """Run both tests for one method against the `none` baseline on the same images."""
    diff, low, high = paired_bootstrap(cms_method, cms_base, n_boot, seed)
    d = np.asarray(per_image_method, float) - np.asarray(per_image_base, float)
    try:
        p = wilcoxon_p(per_image_method, per_image_base)
    except ImportError:  # scipy is not installed on the laptop; it is on Colab
        p = float("nan")
    return Comparison(diff, low, high, float(np.nanmean(d)), p, int(cms_method.shape[0]))
