import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.naive import naive_last_value, seasonal_naive, moving_average  # noqa: E402
from src.evaluation.metrics import wmape, bias, fva  # noqa: E402


def _toy_long_df():
    dates = pd.date_range("2016-01-01", periods=28, freq="D")
    rows = []
    for series_id, level in [("A", 10.0), ("B", 100.0)]:
        for i, d in enumerate(dates):
            rows.append({"id": series_id, "date": d, "sales": level + (i % 7)})
    return pd.DataFrame(rows)


def test_naive_last_value_uses_last_observation_per_series():
    df = _toy_long_df()
    train = df[df["date"] < "2016-01-15"]
    test = df[df["date"] >= "2016-01-15"]
    preds = naive_last_value(train, test)
    last_a = train[train["id"] == "A"].sort_values("date")["sales"].iloc[-1]
    a_mask = (test["id"] == "A").to_numpy()
    assert np.allclose(preds.to_numpy()[a_mask], last_a)


def test_seasonal_naive_recovers_known_weekly_pattern():
    # construct a perfectly repeating weekly pattern and confirm the
    # seasonal-naive prediction matches it exactly (no noise injected)
    dates = pd.date_range("2016-01-04", periods=56, freq="D")  # starts on a Monday
    pattern = [5, 5, 5, 5, 5, 20, 20]  # weekday=5, weekend=20
    sales = [pattern[d.dayofweek] for d in dates]
    df = pd.DataFrame({"id": "A", "date": dates, "sales": sales})
    train = df.iloc[:28]
    test = df.iloc[28:]
    preds = seasonal_naive(train, test)
    expected = [pattern[d.dayofweek] for d in test["date"]]
    assert np.allclose(preds.to_numpy(), expected)


def test_seasonal_naive_beats_naive_last_value_on_seasonal_data():
    dates = pd.date_range("2016-01-04", periods=56, freq="D")
    pattern = [5, 5, 5, 5, 5, 20, 20]
    sales = [pattern[d.dayofweek] for d in dates]
    df = pd.DataFrame({"id": "A", "date": dates, "sales": sales})
    train, test = df.iloc[:28], df.iloc[28:]

    pred_seasonal = seasonal_naive(train, test)
    pred_last = naive_last_value(train, test)

    err_seasonal = wmape(test["sales"], pred_seasonal)
    err_last = wmape(test["sales"], pred_last)
    assert err_seasonal < err_last  # this is exactly the gap Phase 5 is meant to demonstrate


def test_moving_average_smooths_between_extremes():
    df = _toy_long_df()
    train = df[df["date"] < "2016-01-15"]
    test = df[df["date"] >= "2016-01-15"]
    preds = moving_average(train, test, window=7)
    assert (preds > 0).all()


def test_wmape_handles_zero_actuals_without_error():
    y = pd.Series([0, 0, 5, 0])
    p = pd.Series([1, 0, 4, 2])
    result = wmape(y, p)
    assert np.isfinite(result)
    assert result > 0


def test_bias_sign_convention():
    y = pd.Series([10, 10, 10])
    p_over = pd.Series([12, 12, 12])
    p_under = pd.Series([8, 8, 8])
    assert bias(y, p_over) > 0
    assert bias(y, p_under) < 0


def test_fva_positive_when_model_beats_baseline():
    assert fva(baseline_metric=0.20, model_metric=0.15) > 0
    assert fva(baseline_metric=0.20, model_metric=0.25) < 0
