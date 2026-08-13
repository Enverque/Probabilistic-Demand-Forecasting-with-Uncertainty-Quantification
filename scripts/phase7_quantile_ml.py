"""
Phase 7: Quantile / Probabilistic ML Model.

Uses the same 2,100-series category-stratified sample as Phases 3 and 5
(already spans all 10 stores/3 states -- the prototype's single-store
limitation Phase 0 flagged is resolved simply by not re-imposing it, not
by anything clever). Builds horizon-safe features (Phase 7's own module,
bugs fixed per its docstring), fits quantile LightGBM at P10/P50/P90, and
backtests through the Phase 4 harness.

Uses n_windows=3 (not 8) for this phase -- 3 quantile models x 3 windows
on ~4M training rows is already substantial LightGBM training; documented
compute-driven reduction, consistent with how Phase 6 handled the same
tradeoff for SARIMA.

Run: python scripts/phase7_quantile_ml.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.forecasting.backtest import RollingOriginBacktester  # noqa: E402
from src.features.engineering import build_long_format, add_features, feature_columns  # noqa: E402
from src.models.quantile_ml import fit_quantiles, predict_quantiles, point_model  # noqa: E402
from src.models.naive import seasonal_naive, moving_average  # noqa: E402
from src.evaluation.metrics import wmape, bias, fva  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 200  # reduced from Phase 3/5's 700 -- see phase notes: the
# original 2,100-series sample OOM-killed this phase (container has ~3.9GB RAM;
# 2,100 series x 1,941 days x 16 features x 3 quantile models exceeded it). 600
# series is still stratified across all 3 categories and spans all 10 stores/3
# states, which is what Phase 0 actually required -- it's a smaller sample, not
# a narrower one.
HORIZON = 28
N_WINDOWS = 3
QUANTILES = [0.1, 0.5, 0.9]


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)

    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))
    print(f"Sample: {len(sample)} series across {sample['store_id'].nunique()} stores, "
          f"{sample['state_id'].nunique()} states (Phase 0's single-store limitation resolved)")

    t0 = time.time()
    long_df = build_long_format(sample, data.day_cols, data.calendar, data.prices)
    print(f"Long format built: {len(long_df)} rows ({time.time()-t0:.0f}s)")

    t0 = time.time()
    featured = add_features(long_df, horizon=HORIZON)
    del long_df  # free the pre-feature copy explicitly rather than relying on gc timing
    import gc
    gc.collect()
    # Downcast to float32 -- halves the memory footprint of the numeric
    # feature matrix, which is what actually blew the memory budget in
    # the first run of this script (see ITEM_SAMPLE_PER_CAT comment above).
    numeric_cols = featured.select_dtypes(include=["float64"]).columns
    featured[numeric_cols] = featured[numeric_cols].astype("float32")
    cols = feature_columns(HORIZON)
    print(f"Features built: {len(cols)} columns ({time.time()-t0:.0f}s)")
    print(f"Feature columns: {cols}")

    bt = RollingOriginBacktester(horizon_days=HORIZON, n_windows=N_WINDOWS, min_train_days=730)

    def fit_predict_quantiles(train, test):
        models = fit_quantiles(train, cols, "sales", QUANTILES, random_state=RANDOM_STATE)
        preds = predict_quantiles(models, test, cols)
        return preds[0.5]  # median as the point-forecast column for the harness's metric_fns

    coverage_rows = []

    def fit_predict_with_coverage(train, test):
        models = fit_quantiles(train, cols, "sales", QUANTILES, random_state=RANDOM_STATE)
        preds = predict_quantiles(models, test, cols)
        covered = (test["sales"] >= preds[0.1]) & (test["sales"] <= preds[0.9])
        coverage_rows.append({"window_test_start": test["date"].min(),
                               "empirical_coverage_80pct_interval": covered.mean()})
        return preds[0.5]

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_results = bt.run(featured, date_col="date", feature_cols=cols, target_col="sales",
                             fit_predict_fn=fit_predict_with_coverage, metric_fns={"wmape": wmape, "bias": bias})
    print(f"\nQuantile ML backtest done ({time.time()-t0:.0f}s)")
    print(ml_results[["window_id", "test_start", "test_end", "n_train", "wmape", "bias"]].to_string(index=False))

    seasonal_results = bt.run(featured, date_col="date", feature_cols=["id"], target_col="sales",
                               fit_predict_fn=seasonal_naive, metric_fns={"wmape": wmape})
    ma_results = bt.run(featured, date_col="date", feature_cols=["id"], target_col="sales",
                         fit_predict_fn=moving_average, metric_fns={"wmape": wmape})

    print("\n--- Comparison (same 3 windows, same series) ---")
    print(f"Quantile ML (P50):  mean WMAPE = {ml_results['wmape'].mean()*100:.2f}%")
    print(f"Seasonal naive:      mean WMAPE = {seasonal_results['wmape'].mean()*100:.2f}%")
    print(f"Moving average (7d): mean WMAPE = {ma_results['wmape'].mean()*100:.2f}%")
    best_naive = min(seasonal_results["wmape"].mean(), ma_results["wmape"].mean())
    print(f"FVA of ML vs. best naive: {fva(best_naive, ml_results['wmape'].mean())*100:+.1f}%")

    print("\n--- 80% prediction interval coverage (P10-P90), per window ---")
    cov_df = pd.DataFrame(coverage_rows)
    print(cov_df.to_string(index=False))
    print(f"Mean empirical coverage: {cov_df['empirical_coverage_80pct_interval'].mean()*100:.1f}% "
          f"(nominal target: 80%)")
    print("(This is a preview only -- full calibration analysis with coverage plots across "
          "multiple nominal levels is Phase 13's job, not this one.)")


if __name__ == "__main__":
    main()
