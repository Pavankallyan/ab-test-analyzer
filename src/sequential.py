"""Group-sequential testing with Pocock alpha-spending boundaries.

The classic mistake in A/B testing is *peeking*: checking the p-value every
day and stopping when it dips below 0.05 inflates the false-positive rate far
above 5%. Group-sequential designs fix this by pre-specifying K interim
"looks" and testing each against a stricter, adjusted boundary that keeps the
*overall* Type I error at alpha.

This module implements Pocock boundaries: a single constant critical value
``c`` such that

    P(reject H0 at any of the K looks) = alpha,

computed by numerical integration over the joint distribution of the
sequential z-statistics (which under H0 have
corr(Z_j, Z_k) = sqrt(min(j,k) / max(j,k)) for equally spaced looks).
Sanity check: K=5, alpha=0.05 gives c ≈ 2.4128, the textbook Pocock value.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _rejection_probability(c: float, K: int, n_quad: int = 256) -> float:
    """P(|Z_1| >= c or ... or |Z_K| >= c) under H0, via recursive integration.

    Uses the Markov property Z_k | Z_{k-1} = z ~ N(sqrt((k-1)/k)·z, 1/k) and
    propagates the truncated density forward with Gauss-Legendre quadrature
    on (-c, c), which is highly accurate for these smooth integrands.
    """
    x, w = np.polynomial.legendre.leggauss(n_quad)
    z = c * x  # quadrature nodes mapped onto (-c, c)
    qw = w * c  # quadrature weights including the dx/dξ Jacobian

    f = norm.pdf(z)  # density of Z_1, truncated to (-c, c) by the domain
    for k in range(2, K + 1):
        a = np.sqrt((k - 1) / k)
        sd = 1.0 / np.sqrt(k)
        # Transition kernel: p(z_k | z_{k-1}) evaluated on the grid.
        kernel = norm.pdf((z[:, None] - a * z[None, :]) / sd) / sd
        f = kernel @ (qw * f)  # quadrature of ∫ kernel(z_k, z') f(z') dz'

    non_reject = float(np.sum(qw * f))
    return max(0.0, min(1.0, 1.0 - non_reject))


def pocock_boundary(alpha: float = 0.05, K: int = 5) -> float:
    """Constant Pocock critical value for K equally spaced looks at level alpha.

    Found by bisection so that the overall Type I error equals alpha.
    """
    if K < 1:
        raise ValueError("K must be >= 1")
    if K == 1:
        return float(norm.ppf(1.0 - alpha / 2.0))

    lo, hi = norm.ppf(1.0 - alpha / 2.0), 6.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _rejection_probability(mid, K) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def group_sequential_analysis(
    z_stats: list[float],
    alpha: float = 0.05,
) -> dict:
    """Run a Pocock group-sequential analysis over cumulative z-statistics.

    Parameters
    ----------
    z_stats : list of float
        Cumulative z-statistic at each interim look (look 1, look 2, ...).
    alpha : float
        Overall significance level.

    Returns
    -------
    dict with the Pocock boundary, per-look z / p-like comparison, the first
    look at which the trial could have stopped early (or None), and the final
    decision.
    """
    K = len(z_stats)
    boundary = pocock_boundary(alpha, K)
    looks = []
    stop_look = None
    for i, z in enumerate(z_stats, start=1):
        crossed = bool(abs(z) >= boundary)
        if crossed and stop_look is None:
            stop_look = i
        looks.append(
            {"look": i, "z_stat": float(z), "boundary": boundary, "crossed": crossed}
        )
    return {
        "method": "pocock",
        "K": K,
        "alpha": alpha,
        "boundary": boundary,
        "looks": looks,
        "stop_look": stop_look,
        "reject_null": stop_look is not None,
    }
