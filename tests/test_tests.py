"""Unit tests for src/tests.py — hypothesis tests for A/B experiments."""
from __future__ import annotations

import numpy as np
import pytest

from src.tests import mann_whitney_u, two_proportion_ztest, welch_t_test

# ---------------------------------------------------------------------------
# two_proportion_ztest
# ---------------------------------------------------------------------------


class TestTwoProportionZtest:
    def test_detects_clear_lift(self):
        res = two_proportion_ztest(120, 1_000, 80, 1_000)
        assert res["p_value"] < 0.05
        assert res["statistic"] > 0
        assert res["difference"] == pytest.approx(0.04)
        assert res["ci_low"] > 0

    def test_no_difference_gives_high_p(self):
        res = two_proportion_ztest(100, 1_000, 100, 1_000)
        assert res["p_value"] == pytest.approx(1.0, abs=1e-6)
        assert res["difference"] == pytest.approx(0.0)

    def test_result_schema(self, conversions):
        test, control = conversions
        res = two_proportion_ztest(int(test.sum()), len(test),
                                  int(control.sum()), len(control))
        for key in ("test", "n_test", "n_control", "rate_test", "rate_control",
                    "difference", "statistic", "p_value", "ci_low", "ci_high",
                    "alpha"):
            assert key in res, f"missing key: {key}"
        assert res["test"] == "two_proportion_ztest"
        assert res["rate_test"] == pytest.approx(test.mean())
        assert res["ci_low"] < res["difference"] < res["ci_high"]
        assert 0.0 <= res["p_value"] <= 1.0

    def test_sign_flip_reverses_statistic(self):
        a = two_proportion_ztest(120, 1_000, 80, 1_000)
        b = two_proportion_ztest(80, 1_000, 120, 1_000)
        assert a["statistic"] == pytest.approx(-b["statistic"])
        assert a["p_value"] == pytest.approx(b["p_value"])


# ---------------------------------------------------------------------------
# welch_t_test
# ---------------------------------------------------------------------------


class TestWelchTTest:
    def test_detects_mean_lift(self, revenue):
        test, control = revenue
        res = welch_t_test(test, control)
        assert res["p_value"] < 0.01
        assert res["difference"] == pytest.approx(test.mean() - control.mean())
        assert res["ci_low"] > 0

    def test_no_difference_gives_high_p(self, rng):
        a = rng.normal(0, 1, size=500)
        b = rng.normal(0, 1, size=500)
        res = welch_t_test(a, b)
        assert res["p_value"] > 0.1

    def test_accepts_plain_lists(self):
        res = welch_t_test([1.0, 2.0, 3.0, 4.0], [1.1, 2.1, 3.1, 3.9])
        assert res["n_a"] == 4 and res["n_b"] == 4
        assert 0.0 <= res["p_value"] <= 1.0

    def test_result_schema(self, revenue):
        test, control = revenue
        res = welch_t_test(test, control)
        for key in ("test", "n_a", "n_b", "mean_a", "mean_b", "difference",
                    "statistic", "p_value", "df", "ci_low", "ci_high", "alpha"):
            assert key in res, f"missing key: {key}"
        assert res["test"] == "welch_t_test"
        assert res["ci_low"] < res["difference"] < res["ci_high"]
        assert res["df"] > 0


# ---------------------------------------------------------------------------
# mann_whitney_u
# ---------------------------------------------------------------------------


class TestMannWhitneyU:
    def test_detects_shift_on_skewed_data(self, skewed):
        test, control = skewed
        res = mann_whitney_u(test, control)
        assert res["p_value"] < 0.01
        assert res["hl_difference"] < 0  # test latency is lower

    def test_no_difference_gives_high_p(self, rng):
        a = rng.lognormal(0, 0.5, size=400)
        b = rng.lognormal(0, 0.5, size=400)
        res = mann_whitney_u(a, b)
        assert res["p_value"] > 0.05

    def test_ci_brackets_hl_estimate(self, skewed):
        test, control = skewed
        res = mann_whitney_u(test, control)
        assert res["ci_low"] < res["hl_difference"] < res["ci_high"]

    def test_subsampling_path_is_deterministic(self, rng):
        a = rng.normal(5, 2, size=3_000)
        b = rng.normal(5.5, 2, size=3_000)
        first = mann_whitney_u(a, b, ci_subsample=200, seed=123)
        second = mann_whitney_u(a, b, ci_subsample=200, seed=123)
        assert first["ci_subsample_per_group"] == 200
        assert first["hl_difference"] == pytest.approx(second["hl_difference"])
        assert first["ci_low"] == pytest.approx(second["ci_low"])

    def test_result_schema(self, revenue):
        test, control = revenue
        res = mann_whitney_u(test, control)
        for key in ("test", "n_a", "n_b", "statistic", "p_value",
                    "hl_difference", "ci_low", "ci_high",
                    "ci_subsample_per_group", "alpha"):
            assert key in res, f"missing key: {key}"
        assert res["test"] == "mann_whitney_u"
        assert res["n_a"] == len(test) and res["n_b"] == len(control)
