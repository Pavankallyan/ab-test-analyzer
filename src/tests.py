"""Hypothesis tests for A/B experiments.

Three tests, each returning the test statistic, the two-sided p-value, and a
95% (configurable) confidence interval for the difference
(test statistic for the *difference* between groups):

1. Two-proportion z-test — the workhorse for binary conversion outcomes.
2. Welch's t-test — for continuous outcomes without assuming equal variance.
3. Mann-Whitney U — the nonparametric fallback when outcomes are skewed.

All functions are pure: they take arrays/counts in and return result dicts.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from statsmodels.stats.proportion import (
    confint_proportions_2indep,
    proportions_ztest,
)


def two_proportion_ztest(
    successes_test: int,
    n_test: int,
    successes_control: int,
    n_control: int,
    alpha: float = 0.05,
) -> dict:
    """Two-proportion z-test (unpooled variance) for test vs control.

    Returns the z-statistic, two-sided p-value, and a Wald confidence
    interval for the difference in proportions (test − control).
    """
    count = np.array([successes_test, successes_control])
    nobs = np.array([n_test, n_control])
    z_stat, p_value = proportions_ztest(count, nobs, alternative="two-sided")
    ci_low, ci_high = confint_proportions_2indep(
        successes_test, n_test, successes_control, n_control,
        method="wald", alpha=alpha,
    )
    p_test = successes_test / n_test
    p_control = successes_control / n_control
    return {
        "test": "two_proportion_ztest",
        "n_test": n_test,
        "n_control": n_control,
        "rate_test": p_test,
        "rate_control": p_control,
        "difference": p_test - p_control,
        "statistic": float(z_stat),
        "p_value": float(p_value),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "alpha": alpha,
    }


def welch_t_test(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """Welch's t-test for the difference in means (a − b).

    Uses the Welch–Satterthwaite degrees of freedom; the confidence interval
    is the standard t-interval for the difference of means.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = len(a), len(b)
    m1, m2 = a.mean(), b.mean()
    v1, v2 = a.var(ddof=1), b.var(ddof=1)

    t_res = stats.ttest_ind(a, b, equal_var=False, alternative="two-sided")
    se = float(np.sqrt(v1 / n1 + v2 / n2))
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    diff = m1 - m2
    return {
        "test": "welch_t_test",
        "n_a": n1,
        "n_b": n2,
        "mean_a": float(m1),
        "mean_b": float(m2),
        "difference": float(diff),
        "statistic": float(t_res.statistic),
        "p_value": float(t_res.pvalue),
        "df": float(df),
        "ci_low": float(diff - t_crit * se),
        "ci_high": float(diff + t_crit * se),
        "alpha": alpha,
    }


def _hodges_lehmann_ci(
    a: np.ndarray, b: np.ndarray, alpha: float
) -> tuple[float, float]:
    """Distribution-free CI for the median difference (a − b).

    Inverts the Wilcoxon rank-sum test: the Hodges–Lehmann estimator is the
    median of all pairwise differences, and the CI bounds are the k-th and
    (m − k + 1)-th order statistics of those differences, where k comes from
    the normal approximation to the rank-sum distribution
    (Lehmann, *Nonparametrics*, 1975).
    """
    diffs = np.sort((a[:, None] - b[None, :]).ravel())
    m = diffs.size
    n1, n2 = len(a), len(b)
    mu = n1 * n2 / 2.0
    sigma = float(np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0))
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    # Wilcoxon critical value (upper), with continuity correction.
    k = int(np.ceil(mu + 0.5 - z * sigma))
    k = max(1, min(k, m))
    return float(diffs[k - 1]), float(diffs[m - k])


def mann_whitney_u(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float = 0.05,
    ci_subsample: int = 2000,
    seed: int = 0,
) -> dict:
    """Mann-Whitney U test (two-sided) with a Hodges–Lehmann CI.

    The test itself runs on the full samples. The Hodges–Lehmann CI needs all
    pairwise differences (O(n1·n2) memory), so when that exceeds
    ``ci_subsample**2`` pairs each sample is randomly subsampled (fixed seed
    for reproducibility) before computing the CI only.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    u_res = stats.mannwhitneyu(a, b, alternative="two-sided")

    rng = np.random.default_rng(seed)
    a_ci = a if len(a) <= ci_subsample else rng.choice(a, ci_subsample, replace=False)
    b_ci = b if len(b) <= ci_subsample else rng.choice(b, ci_subsample, replace=False)
    # Point estimate on the same subsample used for the CI keeps them coherent.
    hl_point = float(np.median(np.sort((a_ci[:, None] - b_ci[None, :]).ravel())))
    ci_low, ci_high = _hodges_lehmann_ci(a_ci, b_ci, alpha)

    return {
        "test": "mann_whitney_u",
        "n_a": len(a),
        "n_b": len(b),
        "statistic": float(u_res.statistic),
        "p_value": float(u_res.pvalue),
        "hl_difference": hl_point,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_subsample_per_group": len(a_ci),
        "alpha": alpha,
    }
