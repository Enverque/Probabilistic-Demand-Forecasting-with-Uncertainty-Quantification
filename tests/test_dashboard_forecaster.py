import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecasting.dashboard_forecaster import quantile_seasonal_naive  # noqa: E402


def test_quantiles_are_monotonic_at_every_horizon_step():
    rng = np.random.default_rng(0)
    n = 200
    series = rng.poisson(10, n).astype(float)
    dow = np.arange(n + 28) % 7
    result = quantile_seasonal_naive(series, dow, horizon=28, quantiles=[0.1, 0.5, 0.9])
    assert (result[0.1] <= result[0.5]).all()
    assert (result[0.5] <= result[0.9]).all()


def test_output_shape_matches_horizon():
    rng = np.random.default_rng(1)
    n = 150
    series = rng.poisson(5, n).astype(float)
    dow = np.arange(n + 14) % 7
    result = quantile_seasonal_naive(series, dow, horizon=14, quantiles=[0.25, 0.5, 0.75])
    for q in [0.25, 0.5, 0.75]:
        assert len(result[q]) == 14


def test_recovers_known_weekday_pattern():
    n = 140
    dow_hist = np.arange(n) % 7
    pattern = np.array([5, 5, 5, 5, 5, 20, 20])
    series = pattern[dow_hist].astype(float)
    dow_full = np.arange(n + 7) % 7
    result = quantile_seasonal_naive(series, dow_full, horizon=7, quantiles=[0.5])
    expected = pattern[dow_full[n:n + 7]]
    assert np.allclose(result[0.5], expected)
