import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecasting.backtest import RollingOriginBacktester, BacktestWindow  # noqa: E402


def test_generate_windows_basic_shape():
    bt = RollingOriginBacktester(horizon_days=28, n_windows=5, min_train_days=100)
    windows = bt.generate_windows(pd.Timestamp("2011-01-01"), pd.Timestamp("2016-06-19"))
    assert len(windows) == 5
    # chronological order
    assert all(windows[i].test_start < windows[i + 1].test_start for i in range(len(windows) - 1))


def test_windows_rejected_when_insufficient_history():
    # ask for way more windows than 1941 days of history at 28-day steps
    # can support with a 1000-day minimum train requirement
    bt = RollingOriginBacktester(horizon_days=28, n_windows=100, min_train_days=1000)
    windows = bt.generate_windows(pd.Timestamp("2011-01-29"), pd.Timestamp("2016-06-19"))
    assert len(windows) < 100  # must have stopped early, not padded with too-short windows
    for w in windows:
        train_days = (w.train_end - pd.Timestamp("2011-01-29")).days
        assert train_days >= 1000


def test_leakage_check_passes_on_correctly_generated_windows():
    bt = RollingOriginBacktester(horizon_days=28, n_windows=6, min_train_days=100)
    windows = bt.generate_windows(pd.Timestamp("2011-01-01"), pd.Timestamp("2016-06-19"))
    checks = RollingOriginBacktester.validate_no_leakage(windows)
    assert all(checks.values())


def test_leakage_check_catches_a_deliberately_broken_window():
    # Manually construct a window where the test period starts BEFORE the
    # train cutoff -- i.e. inject leakage on purpose -- to prove the
    # validator would actually catch a real bug, not just always pass.
    good = BacktestWindow(0, pd.Timestamp("2016-01-01"), pd.Timestamp("2016-01-01"), pd.Timestamp("2016-01-29"))
    leaky = BacktestWindow(1, pd.Timestamp("2016-02-01"), pd.Timestamp("2016-01-15"), pd.Timestamp("2016-02-12"))
    checks = RollingOriginBacktester.validate_no_leakage([good, leaky])
    assert checks["test_after_train_cutoff"] is False


def test_leakage_check_catches_overlapping_test_windows():
    w1 = BacktestWindow(0, pd.Timestamp("2016-01-01"), pd.Timestamp("2016-01-01"), pd.Timestamp("2016-01-29"))
    w2 = BacktestWindow(1, pd.Timestamp("2016-01-15"), pd.Timestamp("2016-01-15"), pd.Timestamp("2016-02-12"))
    checks = RollingOriginBacktester.validate_no_leakage([w1, w2])
    assert checks["test_windows_non_overlapping"] is False


def test_run_end_to_end_with_naive_model_on_synthetic_data():
    dates = pd.date_range("2011-01-01", periods=800, freq="D")
    df = pd.DataFrame({"date": dates, "x": range(800), "y": [d.dayofweek for d in dates]})

    def fit_predict(train, test):
        # trivial "model": predict the train-set target mean for everyone
        return pd.Series(train["y"].mean(), index=test.index)

    def mae(y_true, y_pred):
        return float((y_true - y_pred).abs().mean())

    bt = RollingOriginBacktester(horizon_days=28, n_windows=4, min_train_days=200)
    results = bt.run(df, date_col="date", feature_cols=["x"], target_col="y",
                      fit_predict_fn=fit_predict, metric_fns={"mae": mae})
    assert len(results) > 0
    assert "mae" in results.columns
    assert (results["mae"] >= 0).all()
    # train sizes should be increasing (expanding window, chronological order)
    assert results["n_train"].is_monotonic_increasing


def test_rolling_mode_keeps_train_size_bounded():
    dates = pd.date_range("2011-01-01", periods=1200, freq="D")
    df = pd.DataFrame({"date": dates, "x": range(1200), "y": 1.0})

    def fit_predict(train, test):
        return pd.Series(0.0, index=test.index)

    bt = RollingOriginBacktester(horizon_days=28, n_windows=4, min_train_days=200,
                                  window_mode="rolling", rolling_train_days=365)
    results = bt.run(df, date_col="date", feature_cols=["x"], target_col="y",
                      fit_predict_fn=fit_predict, metric_fns={"mae": lambda a, b: 0.0})
    # rolling window: train size should stay roughly constant (~365 days),
    # NOT grow window over window the way expanding does
    assert results["n_train"].max() - results["n_train"].min() < 5
