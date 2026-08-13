import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.local_global import (  # noqa: E402
    fit_global_point_model, predict_global, fit_local_point_models, predict_local, MIN_LOCAL_TRAIN_ROWS,
)


def _multi_series_df(n_per_series=200, n_series=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_per_series):
        for s in range(n_series):
            level = 10 * (s + 1)
            rows.append({"id": f"S{s}", "x1": t % 7, "x2": s, "y": max(0, level + rng.normal(0, 2))})
    return pd.DataFrame(rows)


def test_global_model_trains_on_all_series_pooled():
    df = _multi_series_df()
    train, test = df.iloc[:800], df.iloc[800:]
    model = fit_global_point_model(train, ["x1", "x2"], "y")
    preds = predict_global(model, test, ["x1", "x2"])
    assert len(preds) == len(test)
    assert (preds >= 0).all()


def test_local_models_fit_one_per_series_with_enough_data():
    df = _multi_series_df(n_per_series=200)
    train, test = df.iloc[:800], df.iloc[800:]
    models = fit_local_point_models(train, ["x1", "x2"], "y")
    assert len(models) == 5  # one per series
    # every series here has plenty of rows, so none should be a float fallback
    assert all(not isinstance(m, float) for m in models.values())


def test_local_model_falls_back_when_target_is_all_zero():
    # Regression test for a real crash found at M5 scale: LightGBM's
    # Poisson objective errors on an all-zero target ("sum of labels is
    # zero") rather than degrading gracefully. On genuinely intermittent
    # data this is an expected occurrence, not a rare edge case, and must
    # fall back rather than crash the whole backtest.
    df = pd.DataFrame({"id": ["ALLZERO"] * 40, "x1": range(40), "x2": [1] * 40, "y": [0.0] * 40})
    models = fit_local_point_models(df, ["x1", "x2"], "y")  # must not raise
    assert isinstance(models["ALLZERO"], float)
    assert models["ALLZERO"] == 0.0


def test_local_model_falls_back_to_constant_for_cold_start_series():
    df = _multi_series_df(n_per_series=200, n_series=3)
    # simulate a genuinely new series with almost no history
    cold_rows = pd.DataFrame({"id": ["NEW_ITEM"] * 5, "x1": [1, 2, 3, 4, 5],
                               "x2": [99] * 5, "y": [3.0, 4.0, 2.0, 5.0, 3.0]})
    train = pd.concat([df, cold_rows], ignore_index=True)
    models = fit_local_point_models(train, ["x1", "x2"], "y")
    assert isinstance(models["NEW_ITEM"], float)  # too few rows -> fallback constant
    assert abs(models["NEW_ITEM"] - cold_rows["y"].mean()) < 1e-6
    # existing well-populated series should NOT have fallen back
    assert not isinstance(models["S0"], float)


def test_local_prediction_uses_fallback_and_flags_it():
    df = _multi_series_df(n_per_series=200, n_series=2)
    cold_rows = pd.DataFrame({"id": ["NEW_ITEM"] * 3, "x1": [1, 2, 3], "x2": [5, 5, 5], "y": [1.0, 2.0, 1.0]})
    train = pd.concat([df, cold_rows], ignore_index=True)
    test = pd.DataFrame({"id": ["NEW_ITEM", "S0"], "x1": [1, 1], "x2": [5, 0]})
    models = fit_local_point_models(train, ["x1", "x2"], "y")
    preds = predict_local(models, test, ["x1", "x2"])
    fallback_flags = preds.attrs["used_fallback"]
    assert fallback_flags.loc[test["id"] == "NEW_ITEM"].all()
    assert not fallback_flags.loc[test["id"] == "S0"].all()


def test_global_model_can_predict_for_series_with_almost_no_history_using_shared_features():
    # The actual point of Phase 10: a global model trained across many
    # series can still produce a reasonable prediction for a low-history
    # series by leveraging shared feature structure (x2 here stands in for
    # a category/dept code), even though a LOCAL model for that same
    # series has nowhere near enough data to fit at all.
    df = _multi_series_df(n_per_series=200, n_series=4)  # x2 in {0,1,2,3}, y level = 10*(x2+1)
    cold_rows = pd.DataFrame({"id": ["NEW_ITEM"] * 3, "x1": [1, 2, 3], "x2": [1, 1, 1], "y": [21.0, 19.0, 20.5]})
    train = pd.concat([df, cold_rows], ignore_index=True)
    test = pd.DataFrame({"id": ["NEW_ITEM"], "x1": [1], "x2": [1]})

    global_model = fit_global_point_model(train, ["x1", "x2"], "y")
    global_pred = predict_global(global_model, test, ["x1", "x2"]).iloc[0]

    local_models = fit_local_point_models(train, ["x1", "x2"], "y")
    assert isinstance(local_models["NEW_ITEM"], float)  # confirmed: local can't fit meaningfully
    # global model, having seen x2=1 -> y~20 many times across S1, should
    # land in a sane range even for the new item, not near 0 or wildly off
    assert 10 < global_pred < 30
