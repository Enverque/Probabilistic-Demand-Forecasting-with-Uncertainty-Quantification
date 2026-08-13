"""
Phase 4: Time-Series Splitting.

Justifies the rolling-origin backtest design against the ACTUAL 1,941-day
M5 history (not decided in the abstract), then runs the harness end-to-end
on the real Total-level series with a trivial mean-forecast model — not to
evaluate a real model yet (that's Phase 5), but to prove the harness itself
produces leakage-free, sensible windows on real dates before anything is
built on top of it.

Run: python scripts/phase4_backtest.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy  # noqa: E402
from src.forecasting.backtest import RollingOriginBacktester  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HORIZON = 28


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    hierarchy = build_hierarchy(data.sales, data.day_cols)
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)

    n_days = len(data.day_cols)
    print(f"Sales history: {n_days} days ({cal['date'].min().date()} to {cal['date'].max().date()})")

    # --- Design justification ---
    # With HORIZON=28 and a requirement of at least 2 full years of
    # training history before trusting a window (min_train_days=730 --
    # shorter than that and the model has never even seen a full annual
    # cycle, which Phase 3 showed carries a real, if modest, seasonal
    # signal), how many non-overlapping 28-day windows can we fit?
    max_possible_windows = (n_days - 730) // HORIZON
    print(f"\nWith min_train_days=730 and horizon={HORIZON}: "
          f"at most {max_possible_windows} non-overlapping windows fit in the history.")
    # We don't need to use the theoretical max -- 8 windows = 224 held-out
    # days = about 8 months of test data across the backtest, which is
    # enough to see the model's behavior across different seasons/months
    # without leaving so little training data in the earliest window that
    # its score isn't comparable to the others.
    n_windows = 8
    print(f"Using n_windows={n_windows} ({n_windows * HORIZON} total held-out days across the backtest, "
          f"leaving {max_possible_windows - n_windows} windows of headroom).")

    bt = RollingOriginBacktester(horizon_days=HORIZON, n_windows=n_windows, min_train_days=730)
    windows = bt.generate_windows(cal["date"].min(), cal["date"].max())
    leakage_checks = RollingOriginBacktester.validate_no_leakage(windows)
    print(f"\nGenerated {len(windows)} windows. Leakage checks: {leakage_checks}")
    for w in windows:
        train_days = (w.train_end - cal["date"].min()).days
        print(f"  window {w.window_id}: train up to {w.train_end.date()} ({train_days} days) "
              f"-> test {w.test_start.date()} to {w.test_end.date()}")

    # --- End-to-end proof on real data: Total level, trivial model ---
    total_vals = hierarchy["Level1_total"].iloc[0][data.day_cols].to_numpy(dtype=float)
    df = pd.DataFrame({"date": cal["date"], "x": 0, "y": total_vals})  # x is an unused placeholder feature

    def fit_predict_expanding_mean(train, test):
        return pd.Series(train["y"].mean(), index=test.index)

    def mae(y_true, y_pred):
        return float((y_true - y_pred).abs().mean())

    def wmape(y_true, y_pred):
        return float((y_true - y_pred).abs().sum() / (y_true.abs().sum() + 1e-8))

    results = bt.run(df, date_col="date", feature_cols=["x"], target_col="y",
                      fit_predict_fn=fit_predict_expanding_mean,
                      metric_fns={"mae": mae, "wmape": wmape})
    print("\nEnd-to-end run on real Total-level series (trivial mean-forecast model):")
    print(results.to_string(index=False))
    print(f"\nWMAPE mean across windows: {results['wmape'].mean()*100:.1f}%  "
          f"std: {results['wmape'].std()*100:.1f}pp")
    print("(This number is meaningless as a model score -- it's a trivial baseline used "
          "only to prove the harness runs leakage-free end-to-end on real dates. Phase 5 "
          "builds the actual seasonal-naive baseline this should be compared against.)")


if __name__ == "__main__":
    main()
