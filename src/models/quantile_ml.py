"""
Quantile ML model. Notebook 2 used sklearn's
`HistGradientBoostingRegressor(loss="quantile")` — a reasonable and
correct choice, but Phase 0 named LightGBM with quantile loss as the
preferred candidate specifically, so this reimplements the same idea on
LightGBM rather than carrying the sklearn version forward unexamined.
"""
from __future__ import annotations

from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd


def quantile_model(q: float, random_state: int = 42) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(objective="quantile", alpha=q, learning_rate=0.05,
                              n_estimators=300, max_depth=5, num_leaves=31,
                              min_child_samples=30, random_state=random_state, verbose=-1)


def fit_quantiles(train: pd.DataFrame, feature_cols: List[str], target_col: str,
                   quantiles: List[float], random_state: int = 42) -> Dict[float, lgb.LGBMRegressor]:
    models = {}
    for q in quantiles:
        m = quantile_model(q, random_state=random_state)
        m.fit(train[feature_cols], train[target_col])
        models[q] = m
    return models


def predict_quantiles(models: Dict[float, lgb.LGBMRegressor], test: pd.DataFrame,
                       feature_cols: List[str]) -> pd.DataFrame:
    """
    Predicts each quantile and enforces monotonicity (P10 <= P50 <= P90)
    by sorting across quantiles row-wise -- gradient-boosted quantile
    models are trained independently per quantile and have no built-in
    guarantee they won't cross, so this has to be enforced explicitly
    rather than assumed. Clipped at 0 (sales can't be negative) before
    the monotonicity fix, so clipping can't reintroduce a crossing.
    """
    preds = {}
    for q, m in models.items():
        preds[q] = np.clip(m.predict(test[feature_cols]), 0, None)
    pred_df = pd.DataFrame(preds, index=test.index)
    sorted_cols = sorted(pred_df.columns)
    pred_df[sorted_cols] = np.sort(pred_df[sorted_cols].to_numpy(), axis=1)
    return pred_df


def point_model(random_state: int = 42) -> lgb.LGBMRegressor:
    """Median-equivalent point model, Poisson loss (count-appropriate,
    matching the intermittent-demand finding from Phase 3)."""
    return lgb.LGBMRegressor(objective="poisson", learning_rate=0.05, n_estimators=300,
                              max_depth=5, num_leaves=31, min_child_samples=30,
                              random_state=random_state, verbose=-1)
