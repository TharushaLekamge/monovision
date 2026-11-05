"""Paired comparison of two methods over the same scans/images (per-item metric values)."""
import numpy as np
from scipy import stats


def paired_test(a, b, n_boot=10000, seed=0):
    """a, b: per-item scores of method A and B, same order. Returns the mean difference (a - b), a
    bootstrap 95% CI of it, and p-values from the paired t-test and the Wilcoxon signed-rank test."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape or a.ndim != 1 or len(a) < 2:
        raise ValueError("need two 1-D arrays of the same length >= 2")
    d = a - b
    idx = np.random.default_rng(seed).integers(0, len(d), (n_boot, len(d)))
    lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    nonzero = np.any(d != 0)
    return {"n": len(d), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
            "ci95": (float(lo), float(hi)),
            "p_ttest": float(stats.ttest_rel(a, b).pvalue) if nonzero else 1.0,
            "p_wilcoxon": float(stats.wilcoxon(a, b).pvalue) if nonzero else 1.0}
