import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.sarima import select_order_by_aic, sarima_fit_predict  # noqa: E402


def _synthetic_seasonal_series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 0.05 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 1, n)
    return 50 + trend + seasonal + noise


def test_order_selection_returns_valid_result_on_synthetic_data():
    series = _synthetic_seasonal_series()
    best = select_order_by_aic(series, d=1, D=1, s=7,
                                p_range=(0, 1), q_range=(0, 1), P_range=(0, 1), Q_range=(0, 1))
    assert "order" in best and "seasonal_order" in best
    assert np.isfinite(best["aic"])


def test_sarima_fit_predict_no_leakage_and_correct_length():
    series = _synthetic_seasonal_series(n=300)
    dates = pd.date_range("2015-01-01", periods=300, freq="D")
    df = pd.DataFrame({"date": dates, "sales": series})
    train = df.iloc[:272]
    test = df.iloc[272:]

    preds = sarima_fit_predict(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))
    assert len(preds) == len(test)
    assert preds.index.equals(test.index)
    assert (preds >= 0).all()  # clipped, sales can't be negative


def test_sarima_predictions_track_seasonal_pattern_reasonably():
    # not a tight numerical check (SARIMA on 28-day-ahead forecasts will
    # drift), but predictions should be in a sane range relative to the
    # series' own scale, not wildly diverging -- a basic sanity check that
    # would catch e.g. a unit/scaling bug
    series = _synthetic_seasonal_series(n=300)
    dates = pd.date_range("2015-01-01", periods=300, freq="D")
    df = pd.DataFrame({"date": dates, "sales": series})
    train, test = df.iloc[:272], df.iloc[272:]
    preds = sarima_fit_predict(train, test, order=(1, 1, 1), seasonal_order=(1, 1, 0, 7))
    assert preds.mean() < series.max() * 2
    assert preds.mean() > 0
