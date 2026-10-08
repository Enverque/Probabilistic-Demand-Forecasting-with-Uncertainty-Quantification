# Probabilistic Demand Forecasting with Uncertainty Quantification

A retail demand forecasting system for the M5 dataset (Walmart, 30,490
item-store series) that predicts distributions, not just point values, and
connects those distributions to an actual inventory decision.

## Why point forecasts are not enough
A forecast of "542 units" doesn't tell an inventory planner how much
safety stock to hold. This project measures, rather than assumes, how much
better a predictive distribution serves that decision -- see
`docs/FINAL_BUSINESS_ANALYSIS.md` for the answer with real numbers
(10.6-26.5% inventory cost reduction, Phase 16).

## What's actually in here
- Real M5 data (30,490 series, verified against official statistics --
  `docs/phase_notes/phase1.md`)
- 12-level hierarchy construction, verified against the official WRMSSE
  weights file (Phase 2)
- Time-series EDA: intermittency (68% zero-sales days), trend, seasonality,
  structural breaks (Phases 1, 3, 15)
- Rolling-origin backtesting with leakage validation, not `train_test_split`
  (Phase 4)
- Naive baselines, SARIMA (scoped to aggregated levels only, with a
  documented reason it wasn't applied at item level), and quantile
  LightGBM, all backtested and compared on equal footing (Phases 5-7)
- Full probabilistic output (P2.5-P97.5), WRMSSE and pinball loss
  implemented from their definitions and hand-validated, real calibration
  analysis (Phases 8, 11-13)
- Hierarchical reconciliation measured, not assumed to help -- it didn't,
  by 58% (Phase 14)
- A newsvendor inventory simulation connecting forecasts to real cost
  outcomes (Phase 16)
- A Streamlit dashboard, with an explicit scope caveat about which model
  it actually uses (Phase 18)
- 113 automated tests, a lightweight experiment log, and a documented
  reproducibility audit (Phases 19-21)

## Results (see `data/interim/phase22_comparison_table.csv` for the full table)
- Best point-accuracy model: quantile LightGBM, +8.6% FVA over the best
  naive baseline (600-series sample, 3 backtest windows)
- Real, if imperfect, calibration: 50%/80%/95% nominal intervals showed
  74.3%/91.7%/97.5% empirical coverage
- Probabilistic inventory policy beat a point-forecast policy by 10.6-26.5%
  on total cost, in two different cost-asymmetry scenarios
- Bottom-up reconciliation achieved perfect coherence but cost 58% in
  accuracy (full 30,490-series scale) -- a real, measured tradeoff, not
  a free win

## What this project is honest about
- Two real, documented bugs were found and fixed during development: a
  cross-series data leakage bug in rolling features (Phase 10), and a
  hierarchy weight-join bug that would have silently corrupted the overall
  WRMSSE (Phase 11).
- The Bayesian structural model (Phase 9) did not converge and was not
  adopted -- reported with its real diagnostics (r_hat=3.04), not hidden.
- Most ML results are on a 600-series sample, not the full hierarchy
  (compute-constrained; documented per-phase, not silently generalized).

## Setup and running
See `REPRODUCIBILITY.md`.

## Full documentation
- `docs/FINAL_BUSINESS_ANALYSIS.md` -- the business questions, answered
- `docs/RESUME_CLAIMS.md` -- vetted claims, and what was deliberately not claimed
- `docs/phase_notes/` -- decision records for specific phases
- `data/interim/experiment_log.csv` -- every measured result, consolidated
