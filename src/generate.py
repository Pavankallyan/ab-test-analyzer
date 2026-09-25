"""Synthetic experiment data generator (seeded).

Generates a control and a test group with a binary conversion outcome and a
KNOWN true relative lift, plus a pre-period covariate per user that is
correlated with the outcome (for CUPED).

Data-generating process
-----------------------
Each user gets a latent score ``u ~ Normal(0, 1)``. The pre-period covariate
is ``pre = 1{u > threshold}`` where the threshold is set so that
``P(pre = 1) == base_rate``. The conversion outcome is drawn as
``Bernoulli(sigmoid(c_group + beta * u))`` where ``c_group`` is calibrated
(by Gauss-Hermite quadrature) so the *marginal* conversion rate is exactly
``base_rate`` for control and ``base_rate * (1 + true_lift)`` for test —
the true lift is exact in expectation. Because ``u`` drives both the
pre-period covariate and the outcome, the covariate is a genuine predictor
of conversion — exactly what CUPED needs.

CLI usage
---------
python -m src.generate --n-per-group 50000 --base-rate 0.12 \\
    --true-lift 0.05 --seed 42 --output data/experiment.csv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _calibrated_intercept(target_rate: float, beta: float) -> float:
    """Intercept c such that E[sigmoid(c + beta·u)] == target_rate, u ~ N(0,1).

    The latent score inflates the marginal conversion rate (Jensen), so the
    raw logit(target_rate) would overshoot. Solved once by bisection using
    Gauss-Hermite quadrature — deterministic, no simulation noise. This makes
    the *observed* control rate match the configured base rate and the true
    relative lift exact in expectation.
    """
    x, w = hermgauss(64)
    nodes = beta * np.sqrt(2.0) * x
    weights = w / np.sqrt(np.pi)

    def marginal(c: float) -> float:
        return float(np.sum(weights * _sigmoid(c + nodes)))

    lo, hi = -10.0, 10.0  # marginal(-10)≈0 < target < 1≈marginal(10)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if marginal(mid) < target_rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def generate_experiment(
    n_per_group: int,
    base_rate: float,
    true_lift: float,
    seed: int,
    beta: float = 1.2,
) -> pd.DataFrame:
    """Generate a synthetic two-group experiment.

    Parameters
    ----------
    n_per_group : int
        Number of users per group (control and test).
    base_rate : float
        Control-group conversion probability (0 < base_rate < 1).
    true_lift : float
        True *relative* lift of the test group, e.g. 0.05 for +5%.
    seed : int
        Random seed for reproducibility.
    beta : float
        Strength of the latent score's effect on conversion; controls how
        correlated the pre-period covariate is with the outcome.

    Returns
    -------
    pd.DataFrame
        Columns: ``user_id``, ``group`` ("control"/"test"), ``converted``
        (0/1), ``pre_covariate`` (0/1 pre-period conversion indicator).
    """
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    if not 0.0 < base_rate < 1.0:
        raise ValueError("base_rate must be in (0, 1)")
    if base_rate * (1.0 + true_lift) >= 1.0:
        raise ValueError("true_lift pushes the test rate above 1")

    rng = np.random.default_rng(seed)

    frames = []
    pre_threshold = norm.ppf(1.0 - base_rate)  # P(u > t) == base_rate

    for idx, (group, target_rate) in enumerate(
        (("control", base_rate), ("test", base_rate * (1.0 + true_lift)))
    ):
        intercept = _calibrated_intercept(target_rate, beta)
        u = rng.standard_normal(n_per_group)
        pre = (u > pre_threshold).astype(np.int64)
        p = _sigmoid(intercept + beta * u)
        converted = rng.binomial(1, p).astype(np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "user_id": np.arange(n_per_group) + idx * n_per_group,
                    "group": group,
                    "converted": converted,
                    "pre_covariate": pre,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic A/B experiment dataset."
    )
    parser.add_argument("--n-per-group", type=int, default=50000)
    parser.add_argument("--base-rate", type=float, default=0.12)
    parser.add_argument("--true-lift", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/experiment.csv")
    args = parser.parse_args()

    df = generate_experiment(
        n_per_group=args.n_per_group,
        base_rate=args.base_rate,
        true_lift=args.true_lift,
        seed=args.seed,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)

    for group in ("control", "test"):
        sub = df[df["group"] == group]
        print(
            f"{group:8s} n={len(sub):6d} "
            f"observed_rate={sub['converted'].mean():.4f}"
        )
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
