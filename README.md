# ab-test-analyzer

A correct A/B test analysis toolkit in Python: hypothesis testing, power
analysis, group-sequential testing, and CUPED variance reduction — with a
fully worked example on synthetic data where the ground truth is known.

I built this because most A/B testing tutorials get the statistics subtly
wrong: they peek at p-values without correction, size tests by gut feel, and
leave free precision on the table by ignoring pre-period data. This project
does each of those things properly, and proves it on data where I know the
true lift (+5%) because I generated it.

## Results from the worked example

Simulated checkout experiment: 50,000 users per group, 12% base conversion
rate, true relative lift **+5%** (seeded, so the ground truth is known).

| Analysis | Result |
|---|---|
| Observed lift | **+5.41%** relative (+0.65 pp), 95% CI **[+2.03%, +8.80%]** |
| Two-proportion z-test | z = +3.131, **p = 1.74e-03** → significant, reject H0 |
| Welch's t-test | t = +3.131, p = 1.74e-03, 95% CI [+0.0024, +0.0106] |
| Mann-Whitney U | p = 1.74e-03 (agrees with the parametric tests) |
| Power check | 47,030/group needed for 80% power; used 50,000 → **82.3% achieved** |
| Pocock sequential (K=5) | boundary \|z\| ≥ 2.4132; **stopped early at look 4 of 5**, saving 20% of the sample |
| CUPED | **8.6% variance reduction**, CI width shrank **4.6%**, adjusted lift +5.38% (p = 1.10e-03) |

The full readout is in [`examples/readout.txt`](examples/readout.txt).

## Approach

**`src/generate.py` — synthetic experiment data (seeded).** Each user gets a
latent score driving both a pre-period covariate and the conversion outcome,
so the covariate genuinely predicts conversion. The treatment effect is baked
in as a known +5% relative lift, with the intercept calibrated by
Gauss-Hermite quadrature so the *marginal* rates are exactly right
(control ≈ 12%, test ≈ 12.6%). CLI-configurable: `--n-per-group`,
`--base-rate`, `--true-lift`, `--seed`.

**`src/tests.py` — hypothesis tests.** Three pure functions, each returning
the statistic, p-value, and a confidence interval for the difference:
- *Two-proportion z-test* (unpooled) — the primary test for binary outcomes.
- *Welch's t-test* — no equal-variance assumption, with the
  Welch–Satterthwaite CI for the mean difference.
- *Mann-Whitney U* — nonparametric check; CI via the Hodges–Lehmann
  estimator (distribution-free inversion of the rank-sum test). Note: for 0/1
  outcomes the HL estimate is degenerate (pairwise diffs are only −1/0/+1),
  so the p-value is the informative part there.

**`src/power.py` — power analysis.** Answers "how many users do I need?"
before the test (47,030/group for 80% power at a +5% MDE here) and "was my
test big enough?" after it (82.3% achieved power). Uses Cohen's h with
statsmodels' `NormalIndPower`.

**`src/sequential.py` — group-sequential testing (Pocock).** This is the
*peeking* fix: checking results daily and stopping at p < 0.05 inflates the
false-positive rate well above 5%. The Pocock design pre-specifies 5 interim
looks and tests each against a stricter constant boundary (|z| ≥ 2.4132),
computed by numerical integration over the joint distribution of the
sequential z-statistics — I verified it reproduces the textbook values
(K=2 → 2.178, K=5 → 2.413, K=10 → 2.555). On the example data the trial
crosses the boundary at look 4, so it could have stopped early with the
overall 5% error rate intact.

**`src/cuped.py` — CUPED variance reduction.** Regress the outcome on the
pre-period covariate (θ fit on control data only, so the treatment can't leak
into it), then analyze `Y − θ(X − X̄)` instead of `Y`. Same lift estimate
(+5.38% vs +5.41% — the adjustment is unbiased), but 8.6% less variance and
a 4.6% narrower confidence interval. That's precision you get for free from
data you already had.

**`run_example.py`** — the end-to-end pipeline: generate → test → power →
sequential → CUPED → prints the experiment readout and saves it to
`examples/readout.txt`.

## Project structure

```
ab-test-analyzer/
├── src/
│   ├── __init__.py
│   ├── generate.py      # seeded synthetic experiment generator (CLI)
│   ├── tests.py         # z-test, Welch's t-test, Mann-Whitney U
│   ├── power.py         # required n and achieved power (Cohen's h)
│   ├── sequential.py    # Pocock group-sequential boundaries
│   └── cuped.py         # CUPED variance reduction
├── data/
│   └── experiment.csv   # generated data (100k rows, gitignored in real use)
├── examples/
│   └── readout.txt      # full experiment readout from the worked example
├── run_example.py       # end-to-end worked example
└── requirements.txt
```

## How to run

```bash
# from this directory, using the project virtualenv
../.venv/bin/python run_example.py

# or generate a custom dataset directly
../.venv/bin/python -m src.generate --n-per-group 20000 --base-rate 0.10 \
    --true-lift 0.08 --seed 123 --output data/custom.csv
```

Requirements: Python 3.10+, `numpy`, `pandas`, `scipy`, `statsmodels`
(see `requirements.txt`).

## Key design decisions

1. **Ground truth is known.** Every method is validated against data I
   generated with a true +5% lift — if a method can't recover a known effect,
   it's broken. The observed +5.41% with CI [+2.03%, +8.80%] covering +5%
   is exactly what a correct pipeline should produce.
2. **θ is fit on control data only.** Fitting CUPED's regression on pooled
   data would let the treatment effect leak into the adjustment; control-only
   fitting keeps the lift estimate unbiased.
3. **Boundaries are computed, not hardcoded.** The Pocock critical value
   comes from numerically integrating the sequential z-statistic's joint
   distribution (Gauss-Legendre quadrature), and I checked it against
   published tables before trusting it.
4. **Pure functions everywhere.** Every analysis function takes arrays in and
   returns a dict of results — no hidden state, no I/O inside the math, easy
   to unit-test and reuse.
5. **The nonparametric test is a check, not the star.** Mann-Whitney confirms
   the parametric result (p = 1.74e-03, same conclusion); with 100k users the
   CLT has the z-test well covered, and I say so in the readout rather than
   hiding the degenerate HL interval.
