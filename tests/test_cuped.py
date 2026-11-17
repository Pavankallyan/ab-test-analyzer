"""Unit tests for src/cuped.py — CUPED variance reduction.

Covers fit_theta (OLS slope incl. control-only masking), adjust (the
Y - theta*(X - Xbar) transform), ztest_diff (normal-approximation test for a
difference in means), and the full cuped_analysis pipeline on synthetic data
with a known ground truth plus the repo's committed worked-example data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cuped import adjust, cuped_analysis, fit_theta, ztest_diff
from src.generate import generate_experiment


# ---------------------------------------------------------------- fit_theta

def test_fit_theta_exact_linear_relation():
    """y = 2x exactly -> theta == 2."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x
    assert fit_theta(y, x) == pytest.approx(2.0)


def test_fit_theta_negative_slope():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, 7.0, 4.0, 1.0])
    assert fit_theta(y, x) == pytest.approx(-3.0)


def test_fit_theta_sample_mask_uses_only_masked_rows():
    """Masked fit must ignore unmasked rows entirely.

    Full data has slope +1, but the masked subset (first three rows) has
    slope -1; the masked fit must return -1.
    """
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 1.0, 0.0, 3.0, 4.0, 5.0])
    mask = np.array([True, True, True, False, False, False])
    assert fit_theta(y, x, sample=mask) == pytest.approx(-1.0)


def test_fit_theta_zero_variance_covariate_raises():
    with pytest.raises(ValueError, match="zero variance"):
        fit_theta(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0]))


def test_fit_theta_accepts_lists():
    assert fit_theta([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


# ------------------------------------------------------------------- adjust

def test_adjust_theta_zero_is_identity():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([0.0, 1.0, 0.0, 1.0])
    np.testing.assert_array_equal(adjust(y, x, 0.0), y)


def test_adjust_exact_values():
    """y - theta*(x - xbar) with theta=1: [1,2,3] - ([0,1,2] - 1) = [2,2,2]."""
    y = np.array([1.0, 2.0, 3.0])
    x = np.array([0.0, 1.0, 2.0])
    np.testing.assert_array_equal(adjust(y, x, 1.0), np.array([2.0, 2.0, 2.0]))


def test_adjust_preserves_mean():
    """Subtracting theta*(x - xbar) cannot move the mean of y."""
    rng = np.random.default_rng(3)
    y = rng.normal(5.0, 2.0, size=500)
    x = rng.normal(0.0, 1.0, size=500)
    assert adjust(y, x, 1.7).mean() == pytest.approx(y.mean())


def test_adjust_reduces_variance_when_covariate_predicts_outcome(rng):
    """With y = x + noise, the adjusted metric must be less noisy than raw y."""
    x = rng.normal(0.0, 1.0, size=2_000)
    y = x + rng.normal(0.0, 0.3, size=2_000)
    theta = fit_theta(y, x)
    assert adjust(y, x, theta).var(ddof=1) < y.var(ddof=1)


# ---------------------------------------------------------------- ztest_diff

def test_ztest_diff_identical_samples():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    r = ztest_diff(a, a)
    assert r["difference"] == pytest.approx(0.0)
    assert r["statistic"] == pytest.approx(0.0)
    assert r["p_value"] == pytest.approx(1.0)
    assert r["ci_low"] < 0.0 < r["ci_high"]


def test_ztest_diff_known_shift(rng):
    a = rng.normal(1.0, 1.0, size=5_000)
    b = rng.normal(0.0, 1.0, size=5_000)
    r = ztest_diff(a, b)
    assert r["difference"] == pytest.approx(1.0, abs=0.05)
    assert r["p_value"] < 1e-10
    assert r["ci_low"] > 0.0  # CI excludes zero: a real shift
    assert r["ci_width"] == pytest.approx(r["ci_high"] - r["ci_low"])


def test_ztest_diff_alpha_changes_ci_width():
    a = np.arange(1.0, 101.0)
    b = np.arange(1.0, 101.0) + 0.5
    narrow = ztest_diff(a, b, alpha=0.10)
    wide = ztest_diff(a, b, alpha=0.01)
    assert narrow["alpha"] == 0.10
    assert wide["alpha"] == 0.01
    assert wide["ci_width"] > narrow["ci_width"]
    # Same point estimate and statistic regardless of alpha.
    assert wide["difference"] == pytest.approx(narrow["difference"])
    assert wide["p_value"] == pytest.approx(narrow["p_value"])


# ------------------------------------------------------------ cuped_analysis

def test_cuped_analysis_on_generated_data_with_known_lift():
    """On seeded data with a true +5% lift, CUPED must shrink the CI without
    moving the lift estimate, and the covariate must predict the outcome."""
    df = generate_experiment(
        n_per_group=4_000, base_rate=0.12, true_lift=0.05, seed=11
    )
    r = cuped_analysis(df)
    assert r["theta"] > 0.0
    assert 0.0 < r["covariate_outcome_correlation"] < 1.0
    assert 0.0 < r["variance_reduction"] < 0.3
    assert 0.0 < r["ci_width_shrink"] < 0.3
    # Same lift, tighter interval: adjustment is unbiased, not a new effect.
    assert r["adjusted"]["difference"] == pytest.approx(
        r["unadjusted"]["difference"], abs=0.01
    )
    assert r["adjusted"]["ci_width"] < r["unadjusted"]["ci_width"]
    assert set(r) == {
        "theta", "covariate_outcome_correlation", "variance_reduction",
        "unadjusted", "adjusted", "ci_width_shrink", "alpha",
    }


def test_cuped_analysis_uncorrelated_covariate_gives_no_reduction():
    """If the covariate carries no signal, CUPED must not invent a reduction."""
    rng = np.random.default_rng(5)
    n = 2_000
    df = pd.DataFrame({
        "group": ["control"] * n + ["test"] * n,
        "converted": rng.binomial(1, 0.12, size=2 * n).astype(float),
        "pre_covariate": rng.binomial(1, 0.12, size=2 * n).astype(float),
    })
    r = cuped_analysis(df)
    assert abs(r["variance_reduction"]) < 0.05
    assert abs(r["covariate_outcome_correlation"]) < 0.1


def test_cuped_analysis_near_perfect_covariate_removes_almost_all_variance():
    """y = 2x + tiny noise: the adjustment should remove ~all variance."""
    rng = np.random.default_rng(9)
    n = 1_000
    x = rng.normal(0.0, 1.0, size=2 * n)
    y = 2.0 * x + rng.normal(0.0, 0.01, size=2 * n)
    df = pd.DataFrame({
        "group": ["control"] * n + ["test"] * n,
        "converted": y,
        "pre_covariate": x,
    })
    r = cuped_analysis(df)
    assert r["variance_reduction"] > 0.9
    assert r["ci_width_shrink"] > 0.5


def test_cuped_analysis_custom_column_names_and_labels():
    df = generate_experiment(
        n_per_group=1_000, base_rate=0.10, true_lift=0.05, seed=4
    )
    df = df.rename(columns={
        "converted": "y", "pre_covariate": "x", "group": "arm",
    })
    df["arm"] = df["arm"].replace({"control": "A", "test": "B"})
    r = cuped_analysis(
        df, outcome_col="y", covariate_col="x", group_col="arm",
        test_label="B", control_label="A",
    )
    assert r["theta"] > 0.0
    assert r["variance_reduction"] > 0.0


def test_cuped_analysis_on_committed_worked_example():
    """The repo's own data/experiment.csv: the README reports 8.6% variance
    reduction and a 4.6% CI-width shrink — re-check both within a loose band."""
    df = pd.read_csv("data/experiment.csv")
    r = cuped_analysis(df)
    assert 0.05 < r["variance_reduction"] < 0.15
    assert 0.02 < r["ci_width_shrink"] < 0.08
    # The adjusted lift stays within 0.5pp of the unadjusted lift.
    assert abs(r["adjusted"]["difference"] - r["unadjusted"]["difference"]) < 0.005
