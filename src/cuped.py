"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

Idea (Deng et al., 2013): much of the noise in an experiment metric is
predictable from data collected *before* the experiment started. Regress the
outcome Y on a pre-period covariate X, then analyze the adjusted metric

    Y_adj = Y − θ·(X − X̄),   with θ = Cov(Y, X) / Var(X),

instead of Y. The adjustment removes predictable variance without touching
the treatment effect (θ is estimated on control data only, so the treatment
cannot leak into it), which shrinks confidence intervals — equivalent to a
larger sample size, for free.

This module works for any numeric outcome (including 0/1 conversions, where
OLS is the standard CUPED choice). The re-analysis is a normal-approximation
z-test for the difference in adjusted means, mirroring the unadjusted test so
the CI-width comparison is apples to apples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def fit_theta(
    y: np.ndarray, x: np.ndarray, sample: np.ndarray | None = None
) -> float:
    """OLS slope of y on x: θ = Cov(Y, X) / Var(X).

    ``sample`` is a boolean mask selecting the rows used to fit θ; by default
    the control group is used so the treatment effect cannot influence θ.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if sample is not None:
        y, x = y[sample], x[sample]
    var_x = x.var(ddof=1)
    if var_x == 0:
        raise ValueError("Covariate has zero variance; cannot fit theta.")
    return float(np.cov(y, x, ddof=1)[0, 1] / var_x)


def adjust(y: np.ndarray, x: np.ndarray, theta: float) -> np.ndarray:
    """CUPED-adjusted metric: Y − θ·(X − X̄)."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return y - theta * (x - x.mean())


def ztest_diff(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> dict:
    """Normal-approximation z-test for the difference in means (a − b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    diff = a.mean() - b.mean()
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    z = diff / se
    p = float(2.0 * norm.sf(abs(z)))
    z_crit = norm.ppf(1.0 - alpha / 2.0)
    return {
        "difference": float(diff),
        "se": se,
        "statistic": float(z),
        "p_value": p,
        "ci_low": float(diff - z_crit * se),
        "ci_high": float(diff + z_crit * se),
        "ci_width": float(2.0 * z_crit * se),
        "alpha": alpha,
    }


def cuped_analysis(
    df: pd.DataFrame,
    outcome_col: str = "converted",
    covariate_col: str = "pre_covariate",
    group_col: str = "group",
    test_label: str = "test",
    control_label: str = "control",
    alpha: float = 0.05,
) -> dict:
    """Full CUPED pipeline: fit θ on control, adjust, re-test, compare.

    Returns θ, the covariate–outcome correlation, variance reduction on
    control (1 − Var(adj)/Var(raw)), the unadjusted and CUPED-adjusted
    z-tests, and the CI-width shrink from the adjustment.
    """
    y = df[outcome_col].to_numpy(dtype=float)
    x = df[covariate_col].to_numpy(dtype=float)
    is_control = df[group_col].to_numpy() == control_label
    is_test = df[group_col].to_numpy() == test_label

    theta = fit_theta(y, x, sample=is_control)
    correlation = float(np.corrcoef(x[is_control], y[is_control])[0, 1])
    y_adj = adjust(y, x, theta)

    var_reduction = 1.0 - y_adj[is_control].var(ddof=1) / y[is_control].var(ddof=1)

    unadjusted = ztest_diff(y[is_test], y[is_control], alpha)
    adjusted = ztest_diff(y_adj[is_test], y_adj[is_control], alpha)
    ci_shrink = 1.0 - adjusted["ci_width"] / unadjusted["ci_width"]

    return {
        "theta": theta,
        "covariate_outcome_correlation": correlation,
        "variance_reduction": float(var_reduction),
        "unadjusted": unadjusted,
        "adjusted": adjusted,
        "ci_width_shrink": float(ci_shrink),
        "alpha": alpha,
    }
