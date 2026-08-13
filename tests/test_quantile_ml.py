import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.quantile_ml import fit_quantiles, predict_quantiles, point_model  # noqa: E402


def _synthetic_regression_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.integers(0, 5, n)
    noise = rng.normal(0, 1, n)
    y = np.clip(3 + 2 * x1 + x2 + noise, 0, None)
    df = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    return df


def test_quantile_predictions_are_monotonic_after_enforcement():
    df = _synthetic_regression_data()
    train, test = df.iloc[:400], df.iloc[400:]
    models = fit_quantiles(train, ["x1", "x2"], "y", quantiles=[0.1, 0.5, 0.9])
    preds = predict_quantiles(models, test, ["x1", "x2"])
    assert (preds[0.1] <= preds[0.5]).all()
    assert (preds[0.5] <= preds[0.9]).all()


def test_quantile_predictions_are_non_negative():
    df = _synthetic_regression_data()
    train, test = df.iloc[:400], df.iloc[400:]
    models = fit_quantiles(train, ["x1", "x2"], "y", quantiles=[0.1, 0.5, 0.9])
    preds = predict_quantiles(models, test, ["x1", "x2"])
    assert (preds >= 0).all().all()


def test_median_quantile_reasonably_tracks_target():
    df = _synthetic_regression_data(n=2000)
    train, test = df.iloc[:1600], df.iloc[1600:]
    models = fit_quantiles(train, ["x1", "x2"], "y", quantiles=[0.5])
    preds = predict_quantiles(models, test, ["x1", "x2"])
    mae = (preds[0.5] - test["y"]).abs().mean()
    assert mae < test["y"].std()  # should beat "always predict a constant" by a real margin


def test_wider_quantile_spread_for_higher_variance_subgroup():
    # Construct data where x2's variance is deliberately higher for one
    # category to check the model actually widens its interval there,
    # not just shifts its center.
    rng = np.random.default_rng(1)
    n = 3000
    x2 = rng.integers(0, 2, n)
    noise_scale = np.where(x2 == 0, 0.5, 5.0)
    y = np.clip(10 + rng.normal(0, 1, n) * noise_scale, 0, None)
    df = pd.DataFrame({"x1": rng.normal(0, 1, n), "x2": x2, "y": y})
    train, test = df.iloc[:2400], df.iloc[2400:]
    models = fit_quantiles(train, ["x1", "x2"], "y", quantiles=[0.1, 0.9])
    preds = predict_quantiles(models, test, ["x1", "x2"])
    width = preds[0.9] - preds[0.1]
    test = test.reset_index(drop=True)
    width = width.reset_index(drop=True)
    mean_width_low_var = width[test["x2"] == 0].mean()
    mean_width_high_var = width[test["x2"] == 1].mean()
    assert mean_width_high_var > mean_width_low_var
