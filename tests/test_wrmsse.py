import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.wrmsse import scale_denominator, rmsse, wrmsse_for_level, full_wrmsse  # noqa: E402


def test_scale_denominator_hand_computed():
    # Active period (from first nonzero) = [3, 4, 5]; naive diffs = [1, 1];
    # mean squared diff = (1^2 + 1^2) / 2 = 1.0 -- computed by hand.
    train = np.array([0, 3, 4, 5], dtype=float)
    assert scale_denominator(train) == pytest.approx(1.0)


def test_scale_denominator_ignores_pre_launch_zeros():
    # Same active period as above, just with more leading zeros -- scale
    # must be IDENTICAL, since pre-launch zeros shouldn't count.
    train = np.array([0, 0, 0, 0, 0, 3, 4, 5], dtype=float)
    assert scale_denominator(train) == pytest.approx(1.0)


def test_scale_denominator_none_for_all_zero_series():
    train = np.zeros(50)
    assert scale_denominator(train) is None


def test_scale_denominator_none_for_too_short_active_period():
    train = np.array([0, 0, 5], dtype=float)  # only 1 active day, can't diff
    assert scale_denominator(train) is None


def test_rmsse_hand_computed():
    # Full hand-worked example:
    # train = [0,3,4,5] -> scale = 1.0 (from above)
    # actual = [10, 12], predicted = [9, 13]
    # mse = ((10-9)^2 + (12-13)^2) / 2 = (1 + 1) / 2 = 1.0
    # rmsse = sqrt(mse / scale) = sqrt(1.0 / 1.0) = 1.0
    train = np.array([0, 3, 4, 5], dtype=float)
    actual = np.array([10.0, 12.0])
    predicted = np.array([9.0, 13.0])
    result = rmsse(actual, predicted, train)
    assert result == pytest.approx(1.0)


def test_rmsse_perfect_forecast_is_zero():
    train = np.array([0, 3, 4, 5], dtype=float)
    actual = np.array([10.0, 12.0])
    result = rmsse(actual, actual.copy(), train)
    assert result == pytest.approx(0.0)


def test_rmsse_none_when_scale_undefined():
    train = np.zeros(30)
    result = rmsse(np.array([1.0, 2.0]), np.array([1.0, 2.0]), train)
    assert result is None


def test_wrmsse_for_level_hand_computed():
    # 3 series, weights sum to 1: 0.5, 0.3, 0.2
    # RMSSE values: 1.0, 2.0, 0.5
    # weighted sum = 0.5*1.0 + 0.3*2.0 + 0.2*0.5 = 0.5 + 0.6 + 0.1 = 1.2
    rmsse_vals = pd.Series([1.0, 2.0, 0.5])
    weights = pd.Series([0.5, 0.3, 0.2])
    result = wrmsse_for_level(rmsse_vals, weights)
    assert result == pytest.approx(1.2)


def test_wrmsse_for_level_excludes_undefined_without_renormalizing():
    # series 2 has an undefined (None/NaN) RMSSE -- must be EXCLUDED, and
    # the remaining weights must NOT be renormalized to sum to 1, per the
    # conservative design documented in wrmsse.py.
    rmsse_vals = pd.Series([1.0, np.nan, 0.5])
    weights = pd.Series([0.5, 0.3, 0.2])
    result = wrmsse_for_level(rmsse_vals, weights)
    # only series 0 and 2 contribute: 0.5*1.0 + 0.2*0.5 = 0.6 (NOT renormalized to sum-to-1 weights)
    assert result == pytest.approx(0.6)


def test_full_wrmsse_is_simple_average_across_12_levels():
    level_scores = {f"Level{i}": float(i) for i in range(1, 13)}  # scores 1.0..12.0
    result = full_wrmsse(level_scores)
    assert result == pytest.approx(np.mean(range(1, 13)))


def test_full_wrmsse_raises_if_a_level_is_missing():
    level_scores = {f"Level{i}": float(i) for i in range(1, 12)}  # only 11 levels
    level_scores["Level12"] = float("nan")
    with pytest.raises(ValueError):
        full_wrmsse(level_scores)
