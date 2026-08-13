"""
Phase 6: Classical Statistical Model (SARIMA).

Scoped to Total + 10 Store-level series (11 total), per the justification
in src/models/sarima.py's module docstring. Compares against naive/
seasonal-naive baselines computed at THE SAME aggregation level -- Phase 5's
baseline numbers were computed on item-store-level series and are not
comparable to a Total/Store-level SARIMA score, so those baselines are
recomputed here at the correct level rather than reusing Phase 5's numbers
directly.

Uses n_windows=4 (not the 8 used in Phases 4-5) -- 11 series x 8 windows x
a 36-combination-equivalent-cost fit each window would be needlessly slow
for what this phase needs to demonstrate; 4 windows is enough to check
whether SARIMA's edge (if any) is consistent, not a compute-driven
shortcut taken silently.

Run: python scripts/phase6_sarima.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy  # noqa: E402
from src.forecasting.backtest import RollingOriginBacktester  # noqa: E402
from src.models.sarima import select_order_by_aic, sarima_fit_predict  # noqa: E402
from src.models.naive import seasonal_naive, moving_average  # noqa: E402
from src.evaluation.metrics import wmape, bias, fva  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
N_WINDOWS = 4


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    hierarchy = build_hierarchy(data.sales, data.day_cols)
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)

    # --- Stationarity check (justifies d=1, D=1) ---
    total_vals = hierarchy["Level1_total"].iloc[0][data.day_cols].to_numpy(dtype=float)
    adf_levels = adfuller(total_vals)
    adf_diff = adfuller(np.diff(total_vals))
    adf_seasonal_diff = adfuller(total_vals[7:] - total_vals[:-7])
    print("--- Stationarity (ADF test, Total-level series) ---")
    print(f"Levels:         stat={adf_levels[0]:.2f}, p={adf_levels[1]:.4f}")
    print(f"1st difference: stat={adf_diff[0]:.2f}, p={adf_diff[1]:.4f}")
    print(f"Seasonal diff:  stat={adf_seasonal_diff[0]:.2f}, p={adf_seasonal_diff[1]:.4f}")
    print("-> raw series non-stationary, both differenced series stationary: d=1, D=1 justified.\n")

    # --- Order selection on Total series only (see module docstring for why) ---
    print("--- Order selection (AIC grid search, Total series) ---")
    t0 = time.time()
    best = select_order_by_aic(total_vals, d=1, D=1, s=7,
                                p_range=(0, 1, 2), q_range=(0, 1, 2), P_range=(0, 1), Q_range=(0, 1))
    print(f"Selected order={best['order']}, seasonal_order={best['seasonal_order']}, "
          f"AIC={best['aic']:.1f}  ({time.time()-t0:.0f}s)")
    order, seasonal_order = best["order"], best["seasonal_order"]

    # --- Build long-format data for Total + 10 stores ---
    series_map = {"Total": total_vals}
    for _, row in hierarchy["Level3_store"].iterrows():
        series_map[f"Store_{row['store_id']}"] = row[data.day_cols].to_numpy(dtype=float)

    bt = RollingOriginBacktester(horizon_days=28, n_windows=N_WINDOWS, min_train_days=730)

    rows = []
    for name, vals in series_map.items():
        df = pd.DataFrame({"date": cal["date"], "id": name, "sales": vals})

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sarima_results = bt.run(
                df, date_col="date", feature_cols=[], target_col="sales",
                fit_predict_fn=lambda tr, te: sarima_fit_predict(tr, te, order, seasonal_order),
                metric_fns={"wmape": wmape, "bias": bias})
        seasonal_results = bt.run(df, date_col="date", feature_cols=["id"], target_col="sales",
                                   fit_predict_fn=seasonal_naive, metric_fns={"wmape": wmape, "bias": bias})
        ma_results = bt.run(df, date_col="date", feature_cols=["id"], target_col="sales",
                             fit_predict_fn=moving_average, metric_fns={"wmape": wmape, "bias": bias})

        rows.append({
            "series": name,
            "sarima_wmape": sarima_results["wmape"].mean(),
            "seasonal_naive_wmape": seasonal_results["wmape"].mean(),
            "moving_avg_wmape": ma_results["wmape"].mean(),
            "sarima_bias": sarima_results["bias"].mean(),
        })
        print(f"{name}: SARIMA={sarima_results['wmape'].mean()*100:.1f}%  "
              f"seasonal_naive={seasonal_results['wmape'].mean()*100:.1f}%  "
              f"moving_avg={ma_results['wmape'].mean()*100:.1f}%")

    summary = pd.DataFrame(rows)
    summary["best_naive_wmape"] = summary[["seasonal_naive_wmape", "moving_avg_wmape"]].min(axis=1)
    summary["sarima_fva_vs_best_naive"] = summary.apply(
        lambda r: fva(r["best_naive_wmape"], r["sarima_wmape"]), axis=1)

    print("\n--- Summary ---")
    print(summary[["series", "sarima_wmape", "best_naive_wmape", "sarima_fva_vs_best_naive"]]
          .assign(sarima_wmape=lambda d: (d["sarima_wmape"]*100).round(1),
                  best_naive_wmape=lambda d: (d["best_naive_wmape"]*100).round(1),
                  sarima_fva_vs_best_naive=lambda d: (d["sarima_fva_vs_best_naive"]*100).round(1))
          .to_string(index=False))
    print(f"\nSARIMA beats best naive baseline in {(summary['sarima_fva_vs_best_naive'] > 0).sum()}/11 series")
    print(f"Mean FVA vs best naive: {summary['sarima_fva_vs_best_naive'].mean()*100:+.1f}%")


if __name__ == "__main__":
    main()
