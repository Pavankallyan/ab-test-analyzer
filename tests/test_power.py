"""Unit tests for src/power.py — sample-size and achieved-power math.

Ground truths come from the worked example in the README: at a 12% base
rate with a +5% relative MDE, 80% power needs ~47,030 users per group, and
50,000 per group achieves ~82.3% power.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import power


def test_cohens_h_zero_for_equal_rates():
    assert power.cohens_h(0.12, 0.12) == pytest.approx(0.0)


def test_cohens_h_sign_matches_direction():
    # statsmodels' proportion_effectsize(p1, p2) is positive when p1 > p2
    assert power.cohens_h(0.126, 0.12) > 0
    assert power.cohens_h(0.12, 0.126) < 0


def test_cohens_h_symmetry():
    assert power.cohens_h(0.10, 0.20) == pytest.approx(-power.cohens_h(0.20, 0.10))


def test_required_n_matches_worked_example():
    n = power.required_n_per_group(0.12, 0.05, alpha=0.05, power=0.8)
    assert n == pytest.approx(47_030, rel=0.01)


def test_required_n_falls_as_mde_grows():
    small = power.required_n_per_group(0.12, 0.05)
    large = power.required_n_per_group(0.12, 0.20)
    assert large < small


def test_required_n_falls_as_alpha_grows():
    strict = power.required_n_per_group(0.12, 0.05, alpha=0.01)
    loose = power.required_n_per_group(0.12, 0.05, alpha=0.10)
    assert loose < strict


def test_required_n_rises_with_target_power():
    low = power.required_n_per_group(0.12, 0.05, power=0.5)
    high = power.required_n_per_group(0.12, 0.05, power=0.95)
    assert high > low


def test_required_n_rejects_impossible_mde():
    with pytest.raises(ValueError):
        power.required_n_per_group(0.90, 0.20)  # test rate would exceed 1.0


def test_achieved_power_at_required_n_is_target():
    n = power.required_n_per_group(0.12, 0.05, alpha=0.05, power=0.8)
    assert power.achieved_power(int(round(n)), 0.12, 0.05) == pytest.approx(0.8, abs=0.005)


def test_achieved_power_matches_worked_example():
    assert power.achieved_power(50_000, 0.12, 0.05) == pytest.approx(0.823, abs=0.005)


def test_achieved_power_monotone_in_n():
    ns = [10_000, 25_000, 50_000, 100_000]
    powers = [power.achieved_power(n, 0.12, 0.05) for n in ns]
    assert powers == sorted(powers)
    assert all(0.0 < p < 1.0 for p in powers)


def test_power_curve_shape_and_monotonicity():
    grid = np.array([10_000, 25_000, 50_000, 100_000])
    curve = power.power_curve(grid, 0.12, 0.05)
    assert curve.shape == grid.shape
    assert np.all(np.diff(curve) > 0)
    assert curve[-1] > 0.95
