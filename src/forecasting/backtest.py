"""
Rolling-origin (walk-forward) backtesting.

Generalized from the loop in cell 13 of `m5-demand-planning-with-machine-
learning.ipynb`. That loop was correct in its core logic (train on data up
to a cutoff, test on the following TEST_DAYS, step the cutoff backward) but
was written inline against one specific DataFrame/model/metric combination.
Refactored here into a reusable harness so every later phase (baselines,
SARIMA, quantile ML, DeepAR) calls the same window-generation and leakage-
validation code instead of re-deriving cutoff arithmetic per phase — the
exact thing that would let a subtle off-by-one leak into just one phase's
evaluation without anyone noticing.

Design decisions, made explicit rather than left as embedded magic numbers:
- Window count/spacing: justified against the actual 1,941-day history in
  `scripts/phase4_backtest.py`, not decided in the abstract here.
- Non-overlapping test windows (step = horizon) by default: overlapping
  test windows would let adjacent windows' scores be correlated through
  shared test days, inflating the appearance of stable performance across
  "independent" windows.
- Expanding window is the default train mode (all history up to the
  cutoff), not a fixed trailing window, because with only ~5 years of
  history, throwing away early years to keep a fixed window size trades
  away real signal (yearly seasonality, in particular) for a property
  (constant train-set size) that mostly matters for models where training
  cost scales with data size — not the concern in this phase.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd


@dataclass
class BacktestWindow:
    window_id: int
    train_end: pd.Timestamp     # inclusive
    test_start: pd.Timestamp    # exclusive lower bound (test is train_end < date <= test_end)
    test_end: pd.Timestamp      # inclusive


class RollingOriginBacktester:
    def __init__(self, horizon_days: int = 28, n_windows: int = 8,
                 min_train_days: int = 365, window_mode: str = "expanding",
                 rolling_train_days: int = 730):
        """
        horizon_days: forecast horizon per window (28, matching Phase 0's target).
        n_windows: number of backtest origins.
        min_train_days: reject any window whose training set would be
            shorter than this — an origin too close to the start of history
            wouldn't give the model a fair amount of data and would make
            that window's score not comparable to the others.
        window_mode: "expanding" (train on all history up to cutoff) or
            "rolling" (train on the trailing `rolling_train_days` only).
        """
        if window_mode not in ("expanding", "rolling"):
            raise ValueError("window_mode must be 'expanding' or 'rolling'")
        self.horizon_days = horizon_days
        self.n_windows = n_windows
        self.min_train_days = min_train_days
        self.window_mode = window_mode
        self.rolling_train_days = rolling_train_days

    def generate_windows(self, min_date: pd.Timestamp, max_date: pd.Timestamp) -> List[BacktestWindow]:
        windows = []
        for w in range(self.n_windows):
            test_end = max_date - pd.Timedelta(days=w * self.horizon_days)
            test_start = test_end - pd.Timedelta(days=self.horizon_days)
            train_end = test_start

            if self.window_mode == "rolling":
                train_start = train_end - pd.Timedelta(days=self.rolling_train_days)
            else:
                train_start = min_date

            train_days_available = (train_end - max(train_start, min_date)).days
            if train_days_available < self.min_train_days:
                break  # ran out of usable history; don't silently pad with a too-short window

            windows.append(BacktestWindow(window_id=w, train_end=train_end,
                                           test_start=test_start, test_end=test_end))
        return list(reversed(windows))  # chronological order for reporting

    @staticmethod
    def validate_no_leakage(windows: List[BacktestWindow]) -> Dict[str, bool]:
        """
        Structural leakage checks on the window definitions themselves
        (independent of any model): every test period starts strictly
        after its own train cutoff, and test periods across windows don't
        overlap with each other (each simulated origin scores a disjoint
        slice of calendar time, so windows can be compared without one
        window's "future" being another window's test data).
        """
        checks = {}
        checks["test_after_train_cutoff"] = all(w.test_start >= w.train_end for w in windows)
        checks["test_end_after_test_start"] = all(w.test_end > w.test_start for w in windows)
        sorted_windows = sorted(windows, key=lambda w: w.test_start)
        no_overlap = all(sorted_windows[i].test_end <= sorted_windows[i + 1].test_start
                          for i in range(len(sorted_windows) - 1))
        checks["test_windows_non_overlapping"] = no_overlap
        return checks

    def run(self, df: pd.DataFrame, date_col: str, feature_cols: List[str], target_col: str,
            fit_predict_fn: Callable[[pd.DataFrame, pd.DataFrame], "pd.Series"],
            metric_fns: Dict[str, Callable[[pd.Series, pd.Series], float]]) -> pd.DataFrame:
        """
        fit_predict_fn(train_df, test_df) -> predictions aligned to test_df.index
        metric_fns: name -> callable(y_true, y_pred) -> float
        """
        min_date, max_date = df[date_col].min(), df[date_col].max()
        windows = self.generate_windows(min_date, max_date)
        leakage_checks = self.validate_no_leakage(windows)
        if not all(leakage_checks.values()):
            raise RuntimeError(f"Backtest window leakage check failed: {leakage_checks}")

        rows = []
        for w in windows:
            if self.window_mode == "rolling":
                train_start = w.train_end - pd.Timedelta(days=self.rolling_train_days)
                train = df[(df[date_col] > train_start) & (df[date_col] <= w.train_end)]
            else:
                train = df[df[date_col] <= w.train_end]
            test = df[(df[date_col] > w.test_start) & (df[date_col] <= w.test_end)]
            train = train.dropna(subset=feature_cols)
            test = test.dropna(subset=feature_cols)
            if train.empty or test.empty:
                continue

            preds = fit_predict_fn(train, test)
            row = {"window_id": w.window_id, "train_end": w.train_end.date().isoformat(),
                   "test_start": w.test_start.date().isoformat(), "test_end": w.test_end.date().isoformat(),
                   "n_train": len(train), "n_test": len(test)}
            for name, fn in metric_fns.items():
                row[name] = fn(test[target_col], preds)
            rows.append(row)
        return pd.DataFrame(rows)
