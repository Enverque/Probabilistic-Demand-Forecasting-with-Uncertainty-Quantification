# Phase 9 — Deep Probabilistic Model: not adopted

## Decision
The Bayesian structural time series model (`src/models/bayesian_structural.py`,
adapted from the prototype notebook's PyMC negative-binomial model) is
**not adopted** for this project's final model set (Phase 22).

## Why — two independent reasons
1. **Computational infeasibility at project scale.** This environment has
   exactly 1 CPU core. A single fit (Total-level series, one window,
   reduced settings: draws=500/tune=500/chains=2) took 12.2 minutes.
   Extrapolated: a full 8-window backtest on just that one series would
   take ~98 minutes; running at item-store granularity (30,490 series)
   would take an estimated ~259 CPU-days for a single window alone.
2. **The fit did not converge.** Max r_hat across all parameters was 3.04
   (should be <1.01), with 41 divergences and both chains hitting max tree
   depth. ESS_bulk for `sigma_trend`/`alpha` was as low as 2. A posterior
   this far from convergence cannot be trusted, and any forecast numbers
   pulled from it (reported for completeness in Phase 9, not as validated
   results) are not meaningful comparisons to Phases 6/7's numbers.

## Likely cause of the non-convergence
The model's linear predictor is combined additively on a standardized
scale, then affine-transformed to the Negative Binomial mean and clipped
at 0.1 — a real design choice inherited from the prototype, not redesigned
here since the goal was evaluating the prototype's actual model. This
clipping boundary is a plausible source of the poor sampling geometry
(divergences, max tree depth) but was not fixed, since doing so and still
needing to prove tractability at scale would mean solving two hard
problems to justify a model Phase 0 already said not to force.

## What replaced it
Phase 7's quantile LightGBM already delivers genuine, validated
probabilistic output (91.8% empirical coverage on an 80% nominal interval,
measured across 3 real backtest windows) at a small fraction of the compute
cost, with no convergence question mark.
