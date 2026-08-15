"""
Phase 13: Calibration.

Phase 7 checked 80% interval coverage as a single-level, 3-window preview
(91.8% empirical). This phase does the real version: coverage at THREE
nominal levels (50%/80%/95%) across the SAME 3 backtest windows (n_windows
kept at 3, not the project's usual 8, for the same documented compute
reason as Phase 7 -- 7 quantile models x windows is already substantial
LightGBM training), using the Phase 4 harness's window generation directly
(not its generic `run`, since that assumes a single point-prediction
column; calibration needs the full quantile set per window).

Run: python scripts/phase13_calibration.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.features.engineering import build_long_format, add_features, feature_columns  # noqa: E402
from src.forecasting.backtest import RollingOriginBacktester  # noqa: E402
from src.forecasting.probabilistic import (  # noqa: E402
    generate_probabilistic_forecast, build_intervals, INTERVAL_DEFINITIONS,
)
from src.evaluation.calibration import coverage_rate  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 200
HORIZON = 28
N_WINDOWS = 3


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))
    long_df = build_long_format(sample, data.day_cols, data.calendar, data.prices)
    featured = add_features(long_df, horizon=HORIZON)
    numeric_cols = featured.select_dtypes(include=["float64"]).columns
    featured[numeric_cols] = featured[numeric_cols].astype("float32")
    cols = feature_columns(HORIZON)

    bt = RollingOriginBacktester(horizon_days=HORIZON, n_windows=N_WINDOWS, min_train_days=730)
    min_date, max_date = featured["date"].min(), featured["date"].max()
    windows = bt.generate_windows(min_date, max_date)
    leakage_checks = RollingOriginBacktester.validate_no_leakage(windows)
    if not all(leakage_checks.values()):
        raise RuntimeError(f"Leakage check failed: {leakage_checks}")
    print(f"{len(windows)} windows, leakage checks: {leakage_checks}\n")

    per_window_rows = []
    all_actuals, all_intervals = [], []

    for w in windows:
        train = featured[featured["date"] <= w.train_end].dropna(subset=cols)
        test = featured[(featured["date"] > w.test_start) & (featured["date"] <= w.test_end)].dropna(subset=cols)
        if train.empty or test.empty:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = generate_probabilistic_forecast(train, test, cols, "sales", random_state=RANDOM_STATE)
        intervals = build_intervals(preds)
        test = test.reset_index(drop=True)
        intervals = intervals.reset_index(drop=True)

        row = {"window_id": w.window_id, "test_start": w.test_start.date().isoformat(),
               "n_test": len(test)}
        for name in INTERVAL_DEFINITIONS:
            covered = coverage_rate(test["sales"], intervals[f"{name}_lower"], intervals[f"{name}_upper"])
            row[f"coverage_{name}"] = covered
            row[f"mean_width_{name}"] = intervals[f"{name}_width"].mean()
        per_window_rows.append(row)
        all_actuals.append(test["sales"])
        all_intervals.append(intervals)

    results = pd.DataFrame(per_window_rows)
    print("--- Per-window coverage ---")
    display_cols = ["window_id", "test_start", "n_test"] + [f"coverage_{n}" for n in INTERVAL_DEFINITIONS]
    print(results[display_cols].to_string(index=False))

    print("\n--- Calibration summary: nominal vs. empirical coverage, pooled across all windows ---")
    combined_actual = pd.concat(all_actuals, ignore_index=True)
    combined_intervals = pd.concat(all_intervals, ignore_index=True)
    nominal_levels = {"50%": 0.50, "80%": 0.80, "95%": 0.95}
    calibration_rows = []
    for name, nominal in nominal_levels.items():
        covered = coverage_rate(combined_actual, combined_intervals[f"{name}_lower"],
                                 combined_intervals[f"{name}_upper"])
        empirical = covered
        mean_width = combined_intervals[f"{name}_width"].mean()
        gap = empirical - nominal
        calibration_rows.append({"interval": name, "nominal": nominal, "empirical": empirical,
                                  "gap_pp": gap * 100, "mean_width": mean_width,
                                  "direction": "OVERCOVERED" if gap > 0.02 else
                                               ("UNDERCOVERED" if gap < -0.02 else "well-calibrated")})
    calib_df = pd.DataFrame(calibration_rows)
    print(calib_df.to_string(index=False))

    print(f"\nTotal pooled test observations: {len(combined_actual)}")

    print("\n--- Sharpness vs. calibration tradeoff check ---")
    print("An interval that's simply very wide will show high (over-)coverage trivially. "
          "Checking the 95% interval's coverage isn't near 100% (which would indicate the "
          "intervals are absurdly wide rather than genuinely well-calibrated):")
    row_95 = calib_df[calib_df["interval"] == "95%"].iloc[0]
    if row_95["empirical"] > 0.995:
        print(f"  95% interval empirical coverage is {row_95['empirical']*100:.1f}% -- "
              f"suspiciously close to 100%, suggesting the intervals may be wider than needed.")
    else:
        print(f"  95% interval empirical coverage is {row_95['empirical']*100:.1f}% -- "
              f"not saturated at 100%, so the overcoverage seen isn't just 'make it enormous'.")


if __name__ == "__main__":
    main()
