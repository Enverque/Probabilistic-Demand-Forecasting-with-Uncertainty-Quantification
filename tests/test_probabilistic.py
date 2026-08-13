import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.forecasting.probabilistic import (  # noqa: E402
    generate_probabilistic_forecast, build_intervals, skewness_of_intervals, QUANTILE_LEVELS,
)


def _skewed_synthetic_data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    # log-normal target: right-skewed, like real demand data
    y = np.exp(1 + 0.5 * x1 + rng.normal(0, 0.6, n))
    return pd.DataFrame({"x1": x1, "y": y})


def test_all_seven_quantiles_produced_and_monotonic():
    df = _skewed_synthetic_data()
    train, test = df.iloc[:2400], df.iloc[2400:]
    preds = generate_probabilistic_forecast(train, test, ["x1"], "y")
    assert list(preds.columns) == sorted(QUANTILE_LEVELS)
    arr = preds[sorted(QUANTILE_LEVELS)].to_numpy()
    assert (np.diff(arr, axis=1) >= 0).all()


def test_build_intervals_widths_increase_with_nominal_level():
    df = _skewed_synthetic_data()
    train, test = df.iloc[:2400], df.iloc[2400:]
    preds = generate_probabilistic_forecast(train, test, ["x1"], "y")
    intervals = build_intervals(preds)
    # 95% interval should be wider than 80%, which should be wider than 50%
    assert (intervals["95%_width"] >= intervals["80%_width"]).mean() > 0.9
    assert (intervals["80%_width"] >= intervals["50%_width"]).mean() > 0.9


def test_build_intervals_raises_on_missing_quantile():
    df = _skewed_synthetic_data(n=500)
    train, test = df.iloc[:400], df.iloc[400:]
    # only fit 3 quantiles, not the full 95% set -- should fail loudly, not
    # silently return NaN, if asked to build a 95% interval anyway
    preds = generate_probabilistic_forecast(train, test, ["x1"], "y", quantiles=[0.1, 0.5, 0.9])
    with pytest.raises(KeyError):
        build_intervals(preds)  # default interval_defs need 0.025/0.975 too


def test_skewness_detects_right_skew_on_lognormal_target():
    # A log-normal target is right-skewed by construction; the median
    # should sit closer to the lower bound of a central interval than to
    # the upper bound, giving a positive skewness score.
    df = _skewed_synthetic_data(n=4000)
    train, test = df.iloc[:3200], df.iloc[3200:]
    preds = generate_probabilistic_forecast(train, test, ["x1"], "y")
    intervals = build_intervals(preds)
    skew = skewness_of_intervals(intervals, "80%")
    assert skew.median() > 0.05  # meaningfully positive, not just noise around 0
