# Resume Claims (Vetted)

Rule applied: every claim below traces to a number in
`data/interim/experiment_log.csv` or a specific phase's measured result.
No percentage, coverage figure, or savings number appears here that
wasn't actually produced by running code in this repo.

## Supported claims

- Built an end-to-end probabilistic demand forecasting pipeline (M5 dataset,
  30,490 series) with horizon-safe feature engineering, quantile
  regression (LightGBM), and rolling-origin backtesting.
- Implemented rolling-origin backtesting with explicit leakage validation
  (test-after-train-cutoff and non-overlapping-window checks), catching
  a real cross-series data leakage bug in the feature pipeline in the
  process.
- Trained quantile forecasting models (P2.5-P97.5) achieving +8.6% Forecast
  Value Added over the best naive baseline on a 600-series backtested sample.
- Evaluated forecast calibration empirically across three nominal interval
  levels (50%/80%/95%) over 3 backtest windows, finding consistent but
  bounded overcoverage (2.5-24.3 percentage points, tightest at the tails).
- Implemented WRMSSE and pinball loss from their mathematical definitions,
  each validated against a hand-computed example before use.
- Implemented hierarchical forecast reconciliation (bottom-up, full 30,490-
  series scale) and measured its actual accuracy tradeoff: achieved perfect
  coherence (0.0 gap) at a measured 58% WRMSSE cost versus direct forecasting.
- Translated predictive distributions into inventory decisions via a
  newsvendor model, measuring a 10.6-26.5% cost reduction versus a
  point-forecast ordering policy across two cost-asymmetry scenarios.
- Compared global vs. local forecasting strategies empirically, including
  an engineered cold-start scenario, finding the global model matched or
  outperformed per-series local models on both slices tested.
- Built a working reproducibility and testing discipline: 113 automated
  tests, a documented dependency audit, and a lightweight experiment log
  consolidating all measured results.

## Explicitly NOT claimed, and why

- NOT "improved forecast accuracy by X%" as a headline without qualifying
  it to the specific sample/window scope it was measured on (the project's
  own numbers differ meaningfully between the 600-series sample and full
  30,490-series scale, and conflating them would misrepresent the work).
- NOT "built a production-ready Bayesian forecasting model" -- Phase 9's
  model did not converge and was explicitly not adopted.
- NOT "reconciliation improves forecast accuracy" -- measured the opposite
  (58% worse) for the method tested, and said so.
- NOT any WRMSSE/coverage/cost number for the FULL 30,490-series ML
  pipeline -- ML results in this project are sample-scope (600 series);
  only the naive-baseline and reconciliation results were run at full scale.
