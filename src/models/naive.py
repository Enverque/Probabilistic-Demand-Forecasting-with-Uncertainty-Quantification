"""
Baseline forecasters, adapted from `naive_seasonal()` in notebook 2 to the
`fit_predict_fn(train, test) -> predictions` interface expected by
`RollingOriginBacktester.run()`.

Three baselines, per Phase 0's spec (naive / seasonal naive / moving
average where appropriate):
- naive_last_value: predicts the last observed value per series. The
  correct baseline for a non-seasonal, slow-moving series; will look bad
  here given the strong weekly seasonality Phase 3 found, and that gap
  IS the point of including it.
- seasonal_naive: reused from notebook 2 as-is (mean of the last 4
  same-weekday observations per series).
- moving_average: predicts the trailing 7-day mean per series. A cheap
  smoother; expected to sit between the other two.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def naive_last_value(train: pd.DataFrame, test: pd.DataFrame,
                      id_col: str = "id", date_col: str = "date", target_col: str = "sales") -> pd.Series:
    last_vals = train.sort_values(date_col).groupby(id_col)[target_col].last()
    fallback = train[target_col].mean()
    preds = test[id_col].map(last_vals).fillna(fallback)
    return pd.Series(preds.to_numpy(), index=test.index)


def seasonal_naive(train: pd.DataFrame, test: pd.DataFrame,
                    id_col: str = "id", date_col: str = "date", target_col: str = "sales",
                    n_periods: int = 4) -> pd.Series:
    h = train.copy()
    h["dow"] = h[date_col].dt.dayofweek
    ref = (h.groupby([id_col, "dow"])[target_col]
           .apply(lambda s: s.tail(n_periods).mean())
           .reset_index().rename(columns={target_col: "pred"}))
    t = test.copy()
    t["dow"] = t[date_col].dt.dayofweek
    t = t.merge(ref, on=[id_col, "dow"], how="left")
    t["pred"] = t["pred"].fillna(h[target_col].mean())
    return pd.Series(t["pred"].to_numpy(), index=test.index)


def moving_average(train: pd.DataFrame, test: pd.DataFrame,
                    id_col: str = "id", date_col: str = "date", target_col: str = "sales",
                    window: int = 7) -> pd.Series:
    last_n = (train.sort_values(date_col).groupby(id_col)[target_col]
              .apply(lambda s: s.tail(window).mean()))
    fallback = train[target_col].mean()
    preds = test[id_col].map(last_n).fillna(fallback)
    return pd.Series(preds.to_numpy(), index=test.index)
