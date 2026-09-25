"""Power analysis for A/B experiments.

Two questions every experimenter should answer *before* running a test:

1. "How many users do I need?" — ``required_n_per_group``
2. "Was my test big enough to detect the effect I care about?" —
   ``achieved_power``

Both use Cohen's h (the arcsine-transformed difference of proportions) with
statsmodels' ``NormalIndPower``, the standard normal-approximation
sample-size calculation for two independent proportions.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h effect size between two proportions."""
    return float(proportion_effectsize(p1, p2))


def required_n_per_group(
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Sample size per group to detect a relative lift with given power.

    Parameters
    ----------
    baseline_rate : float
        Expected control conversion rate.
    mde_relative : float
        Minimum detectable effect as a *relative* lift, e.g. 0.05 for +5%.
    alpha : float
        Two-sided significance level.
    power : float
        Desired statistical power (1 − β).

    Returns
    -------
    float
        Required users per group (round up to the next integer).
    """
    p2 = baseline_rate * (1.0 + mde_relative)
    if not 0.0 < p2 < 1.0:
        raise ValueError("MDE pushes the test rate outside (0, 1)")
    h = cohens_h(baseline_rate, p2)
    n = NormalIndPower().solve_power(
        effect_size=h, alpha=alpha, power=power, alternative="two-sided"
    )
    return float(n)


def achieved_power(
    n_per_group: int,
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
) -> float:
    """Power actually achieved at a given sample size for the target MDE."""
    p2 = baseline_rate * (1.0 + mde_relative)
    h = cohens_h(baseline_rate, p2)
    return float(
        NormalIndPower().power(
            effect_size=h, nobs1=n_per_group, alpha=alpha,
            ratio=1.0, alternative="two-sided",
        )
    )


def power_curve(
    n_per_group_values: np.ndarray,
    baseline_rate: float,
    mde_relative: float,
    alpha: float = 0.05,
) -> np.ndarray:
    """Achieved power over a grid of per-group sample sizes (for plotting)."""
    return np.array(
        [
            achieved_power(int(n), baseline_rate, mde_relative, alpha)
            for n in n_per_group_values
        ]
    )
