"""End-to-end worked example: a simulated A/B test with a KNOWN +5% lift.

Pipeline:
  1. Generate synthetic experiment data (seeded, ground truth known).
  2. Run the three hypothesis tests.
  3. Check power: required n for the MDE vs. the n actually used.
  4. Run a Pocock group-sequential analysis over 5 interim looks.
  5. Apply CUPED variance reduction and re-test.
  6. Print a full experiment readout and save it to examples/readout.txt.

Run from the project root:
    .venv/bin/python run_example.py
(or: python run_example.py with the project venv active)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest

from src import cuped, generate, power, sequential, tests

# --- Experiment configuration (ground truth) ---
N_PER_GROUP = 50_000
BASE_RATE = 0.12
TRUE_LIFT = 0.05  # +5% relative lift, known by construction
SEED = 42
K_LOOKS = 5
ALPHA = 0.05

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "experiment.csv")
READOUT_PATH = os.path.join(BASE_DIR, "examples", "readout.txt")


def cumulative_z_stats(df: pd.DataFrame, k: int, seed: int) -> list[float]:
    """Cumulative two-proportion z-statistic at each of k interim looks.

    Users are shuffled once (fixed seed) to simulate arrival order, then the
    data is split into k equal chunks and the z-statistic is recomputed
    cumulatively at each look — exactly how an analyst would peek.
    """
    order = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    idx_chunks = np.array_split(np.arange(len(order)), k)
    z_stats = []
    cum_idx = []
    for idx in idx_chunks:
        cum_idx.append(idx)
        so_far = order.iloc[np.concatenate(cum_idx)]
        t = so_far[so_far["group"] == "test"]
        c = so_far[so_far["group"] == "control"]
        z, _ = proportions_ztest(
            [int(t["converted"].sum()), int(c["converted"].sum())],
            [len(t), len(c)],
            alternative="two-sided",
        )
        z_stats.append(float(z))
    return z_stats


def main() -> None:
    lines: list[str] = []
    out = lines.append

    # 1. Generate ---------------------------------------------------------
    df = generate.generate_experiment(
        n_per_group=N_PER_GROUP,
        base_rate=BASE_RATE,
        true_lift=TRUE_LIFT,
        seed=SEED,
    )
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    df.to_csv(DATA_PATH, index=False)

    control = df[df["group"] == "control"]
    test = df[df["group"] == "test"]
    n_c, n_t = len(control), len(test)
    s_c = int(control["converted"].sum())
    s_t = int(test["converted"].sum())
    rate_c = s_c / n_c
    rate_t = s_t / n_t
    obs_lift = (rate_t - rate_c) / rate_c

    out("=" * 70)
    out("A/B TEST READOUT — simulated checkout experiment")
    out("=" * 70)
    out(f"Design:        control vs test, n={N_PER_GROUP:,}/group, "
        f"base rate={BASE_RATE:.0%}, TRUE lift=+{TRUE_LIFT:.0%} (known, seeded)")
    out("")
    out("-" * 70)
    out("1. OBSERVED RESULTS")
    out("-" * 70)
    out(f"  control: n={n_c:,}, conversions={s_c:,}, rate={rate_c:.4f}")
    out(f"  test:    n={n_t:,}, conversions={s_t:,}, rate={rate_t:.4f}")
    out(f"  observed lift: {obs_lift:+.2%} relative "
        f"({(rate_t - rate_c) * 100:+.2f} pp absolute)")
    out("")

    # 2. Hypothesis tests --------------------------------------------------
    out("-" * 70)
    out("2. HYPOTHESIS TESTS (H0: no difference, alpha=0.05)")
    out("-" * 70)
    z_res = tests.two_proportion_ztest(s_t, n_t, s_c, n_c, ALPHA)
    t_res = tests.welch_t_test(
        test["converted"].to_numpy(), control["converted"].to_numpy(), ALPHA
    )
    mw_res = tests.mann_whitney_u(
        test["converted"].to_numpy(), control["converted"].to_numpy(), ALPHA
    )
    out(f"  two-proportion z-test: z={z_res['statistic']:+.3f}, "
        f"p={z_res['p_value']:.3e}")
    out(f"    lift estimate {(z_res['difference'] / rate_c):+.2%} relative, "
        f"95% CI [{(z_res['ci_low'] / rate_c):+.2%}, {(z_res['ci_high'] / rate_c):+.2%}]")
    out(f"  Welch's t-test:        t={t_res['statistic']:+.3f}, "
        f"p={t_res['p_value']:.3e}, "
        f"95% CI [{t_res['ci_low']:+.4f}, {t_res['ci_high']:+.4f}]")
    out(f"  Mann-Whitney U:        U={mw_res['statistic']:,.0f}, "
        f"p={mw_res['p_value']:.3e}")
    out(f"    Hodges-Lehmann median diff {mw_res['hl_difference']:+.4f}, "
        f"95% CI [{mw_res['ci_low']:+.4f}, {mw_res['ci_high']:+.4f}]")
    out("    (HL is degenerate for 0/1 outcomes: pairwise diffs are only "
        "-1/0/+1, so the median diff is 0 — the p-value is what matters here)")
    sig = "SIGNIFICANT — reject H0" if z_res["p_value"] < ALPHA else "not significant"
    out(f"  => primary test (z-test): {sig}")
    out("")

    # 3. Power -------------------------------------------------------------
    out("-" * 70)
    out("3. POWER ANALYSIS (MDE = +5% relative, alpha=0.05, target power=0.80)")
    out("-" * 70)
    n_req = power.required_n_per_group(BASE_RATE, TRUE_LIFT, ALPHA, 0.80)
    n_ach = power.achieved_power(N_PER_GROUP, BASE_RATE, TRUE_LIFT, ALPHA)
    out(f"  required n per group for 80% power: {n_req:,.0f}")
    out(f"  actual n per group:                 {N_PER_GROUP:,}")
    out(f"  achieved power at actual n:         {n_ach:.1%}")
    out(f"  => the test was "
        f"{'adequately' if n_ach >= 0.8 else 'under-'} powered for the MDE")
    out("")

    # 4. Sequential --------------------------------------------------------
    out("-" * 70)
    out(f"4. GROUP-SEQUENTIAL ANALYSIS (Pocock, K={K_LOOKS} looks, alpha=0.05)")
    out("-" * 70)
    z_looks = cumulative_z_stats(df, K_LOOKS, SEED)
    seq = sequential.group_sequential_analysis(z_looks, ALPHA)
    out(f"  Pocock boundary (constant critical value): "
        f"|z| >= {seq['boundary']:.4f}")
    for look in seq["looks"]:
        flag = " <-- STOP, boundary crossed" if look["crossed"] else ""
        out(f"  look {look['look']}: z={look['z_stat']:+.3f}{flag}")
    if seq["stop_look"] is not None:
        out(f"  => could have stopped early at look {seq['stop_look']} "
            f"of {K_LOOKS} with the overall 5% error rate intact")
    else:
        out("  => never crossed the boundary; no early stopping")
    out("")

    # 5. CUPED -------------------------------------------------------------
    out("-" * 70)
    out("5. CUPED VARIANCE REDUCTION (pre-period covariate)")
    out("-" * 70)
    cu = cuped.cuped_analysis(df)
    ua, ad = cu["unadjusted"], cu["adjusted"]
    out(f"  theta (control fit): {cu['theta']:.4f}")
    out(f"  covariate-outcome correlation: {cu['covariate_outcome_correlation']:+.3f}")
    out(f"  variance reduction on control: {cu['variance_reduction']:.1%}")
    out(f"  unadjusted: lift {(ua['difference'] / rate_c):+.2%}, "
        f"p={ua['p_value']:.3e}, CI width={(ua['ci_width'] / rate_c):.2%} (relative)")
    out(f"  CUPED-adjusted: lift {(ad['difference'] / rate_c):+.2%}, "
        f"p={ad['p_value']:.3e}, CI width={(ad['ci_width'] / rate_c):.2%} (relative)")
    out(f"  CI width shrink from CUPED: {cu['ci_width_shrink']:.1%}")
    out("")

    out("=" * 70)
    out("CONCLUSION")
    out("=" * 70)
    out(f"  The test detected the true +5% lift "
        f"(observed {obs_lift:+.2%}, p={z_res['p_value']:.2e}).")
    if seq["stop_look"] is not None:
        out(f"  A Pocock sequential design would have stopped at look "
            f"{seq['stop_look']}/{K_LOOKS}, saving "
            f"{(K_LOOKS - seq['stop_look']) / K_LOOKS:.0%} of the sample.")
    else:
        out("  No early stopping was warranted under the Pocock design.")
    out(f"  CUPED cut variance by {cu['variance_reduction']:.1%} and shrank "
        f"the CI by {cu['ci_width_shrink']:.1%} — free precision from "
        "pre-period data.")
    out("")

    readout = "\n".join(lines)
    print(readout)
    os.makedirs(os.path.dirname(READOUT_PATH), exist_ok=True)
    with open(READOUT_PATH, "w") as f:
        f.write(readout + "\n")
    print(f"Readout saved to {READOUT_PATH}")


if __name__ == "__main__":
    main()
