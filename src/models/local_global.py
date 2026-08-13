"""
Global (one model, all series pooled) vs. local (one model per series)
point forecasters, sharing the same LightGBM/Poisson-loss family so the
comparison isolates training SCOPE, not model family.

Local models fall back to a simple constant when a series doesn't have
enough training rows to fit meaningfully (see MIN_LOCAL_TRAIN_ROWS) --
this is deliberate and explicit, not a silent failure: it's exactly the
cold-start scenario Phase 10 exists to demonstrate.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from src.models.quantile_ml import point_model

MIN_LOCAL_TRAIN_ROWS = 30  # below this, a per-series LightGBM fit isn't meaningful


def fit_global_point_model(train: pd.DataFrame, feature_cols: List[str], target_col: str,
                            random_state: int = 42):
    model = point_model(random_state=random_state)
    model.fit(train[feature_cols], train[target_col])
    return model


def predict_global(model, test: pd.DataFrame, feature_cols: List[str]) -> pd.Series:
    return pd.Series(np.clip(model.predict(test[feature_cols]), 0, None), index=test.index)


def fit_local_point_models(train: pd.DataFrame, feature_cols: List[str], target_col: str,
                            id_col: str = "id", random_state: int = 42) -> Dict[str, object]:
    """
    Returns id -> fitted model, OR id -> float (a fallback constant, the
    series' own training mean) when there isn't enough data to fit a
    per-series model. The fallback is returned as a plain float rather
    than silently skipped, so callers always get a usable value and can
    still tell which series used the fallback (isinstance check).
    """
    models: Dict[str, object] = {}
    for series_id, g in train.groupby(id_col):
        g_clean = g.dropna(subset=feature_cols)
        # Two conditions require the constant fallback, not one: too few
        # rows to fit meaningfully, OR (found while running this at real
        # M5 scale) a target that sums to zero -- LightGBM's Poisson
        # objective errors outright ("sum of labels is zero") rather than
        # degrading gracefully, and on a dataset with a 68% zero-sales
        # rate (Phase 1), an all-zero local training slice is a real,
        # expected occurrence for a genuinely intermittent series, not an
        # edge case to special-case away. The correct fallback in that
        # situation is exactly the same constant (0) the mean would give,
        # so this doesn't change what a well-fit model would have
        # predicted anyway -- it just avoids the crash.
        if len(g_clean) < MIN_LOCAL_TRAIN_ROWS or g_clean[target_col].sum() == 0:
            models[series_id] = float(g[target_col].mean()) if len(g) else 0.0
            continue
        m = point_model(random_state=random_state)
        m.fit(g_clean[feature_cols], g_clean[target_col])
        models[series_id] = m
    return models


def predict_local(models: Dict[str, object], test: pd.DataFrame, feature_cols: List[str],
                   id_col: str = "id") -> pd.Series:
    preds = pd.Series(0.0, index=test.index)
    used_fallback = pd.Series(False, index=test.index)
    for series_id, g in test.groupby(id_col):
        m = models.get(series_id)
        if m is None:
            preds.loc[g.index] = 0.0
            used_fallback.loc[g.index] = True
        elif isinstance(m, float):
            preds.loc[g.index] = m
            used_fallback.loc[g.index] = True
        else:
            preds.loc[g.index] = np.clip(m.predict(g[feature_cols]), 0, None)
    result = preds
    result.attrs["used_fallback"] = used_fallback
    return result
