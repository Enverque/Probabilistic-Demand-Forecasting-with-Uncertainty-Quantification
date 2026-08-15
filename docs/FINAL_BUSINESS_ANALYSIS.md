# Final Business Analysis

Every answer below cites a specific measured result already logged in
`data/interim/experiment_log.csv` (Phase 21) and detailed in the phase
that produced it. No number here was computed for the first time in this
document.

## Forecasting

**Which model forecasts demand most accurately?**
Quantile LightGBM (global), at item-store level: WMAPE 73.8% vs. 83.1%
for the best naive baseline on the same 600-series sample and windows
(FVA +8.6%, Phase 7). It's the only model in this project with backtested
evidence of beating both naive baselines.

**How much does the ML model improve over naive baselines?**
+8.6% Forecast Value Added over the best naive baseline (moving average),
measured across 3 backtest windows on the 600-series sample (Phase 7).
This is a real but modest improvement, not a dramatic one -- stated
plainly rather than inflated.

## Uncertainty

**Are the prediction intervals calibrated?**
Not perfectly -- consistently overcovered at all three nominal levels
tested (50% -> 74.3% empirical, 80% -> 91.7%, 95% -> 97.5%, pooled across
3 windows, Phase 13). Overcoverage shrinks as the nominal level rises,
consistent with zero-inflation compressing the model's inner quantiles
(Phase 13's interpretation, tied back to Phase 3's 68% zero-rate finding).

**Are they too wide or too narrow?**
Too wide, specifically at the 50% interval (+24.3 percentage points of
overcoverage) -- the 95% interval is close to well-calibrated (+2.5pp).
This means there's real room to sharpen the model's inner quantiles
without sacrificing the already-good tail calibration.

**Which model gives the best calibration/sharpness tradeoff?**
Only the quantile LightGBM model was evaluated for calibration in this
project -- SARIMA's native intervals and the (non-converged) Bayesian
model's intervals were not validated, so this question can only be
answered for one model, and that limitation is stated rather than papered
over.

## Hierarchy

**Does reconciliation improve coherence?**
Yes, completely and by construction -- bottom-up reconciled forecasts
showed a max coherence gap of exactly 0.0000000000 between store-level
sums and state-level totals (Phase 14), versus up to 0.81% gap for
independent, non-reconciled forecasts at the same levels (Phase 2).

**Does it improve accuracy?**
No -- it made accuracy substantially worse: full-scale WRMSSE rose from
0.7795 (direct, independent forecasts per level) to 1.2328 (bottom-up
reconciled), a 58% degradation (Phase 14). This is measured evidence
against assuming reconciliation is a free accuracy win, which Phase 0
explicitly warned not to assume.

## Inventory

**Does probabilistic forecasting reduce expected inventory cost?**
Yes, in both cost scenarios tested: +26.5% cost reduction when stockouts
are 4x more expensive than holding, +10.6% when holding is 3x more
expensive than stockouts (Phase 16, same 600-series sample and quantile
forecasts validated in Phases 7-13).

**Does it improve service level?**
It changes service level in the direction the cost structure actually
calls for, which is the more precise and more useful claim than "improves"
in general: service level rose from 64.8% to 82.6% in the stockout-expensive
scenario, and fell from 64.8% to 59.7% in the holding-expensive scenario
(Phase 16) -- the point-forecast policy produced the identical 64.8% in
both scenarios because it never looks at the cost ratio at all.

**How does it change reorder decisions?**
The probabilistic policy orders more than the point-forecast policy when
stockouts are expensive (mean excess inventory rose from 0.18 to 1.10
units) and less when holding is expensive (mean excess fell from 0.18 to
0.03 units) -- direct evidence that a predictive distribution, not just a
point estimate, is necessary to make a cost-asymmetry-aware ordering
decision at all (Phase 16).
