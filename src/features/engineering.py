"""
Horizon-safe feature engineering for the quantile ML model.

The lag/rolling logic is reused from `add_features()` in notebook 2 — it
was genuinely correct: lag features are only included if `lag >= horizon`
(so a 28-day-ahead forecast can't use a lag_1 feature that wouldn't exist
yet at prediction time), and rolling stats are computed on a series
shifted by `horizon` first, which is the right way to keep a rolling
window from reaching into the forecast period.

Two real bugs found in the surrounding code and fixed here rather than
carried forward:

1. `snap_CA` was hardcoded regardless of which state a store belongs to.
   The prototype only ever ran on one (California) store, so this never
   surfaced. Extended to multiple stores/states here (Phase 0 explicitly
   asked for this), so it's fixed: each row now gets the SNAP flag for
   its OWN state, via a per-state lookup.
2. `promo_flag` was `event_name_1.notna()` — that's an event/holiday
   indicator, not a promotion indicator. M5 has no real promotion or
   discount field at all (its price table gives sell price, not a
   promo flag or original price to compare against). Renamed to
   `is_event` to describe what it actually is, and NOT presented to the
   model as a promo signal it isn't.

New, not in the prototype: price-based features (relative price vs. the
item's own trailing average, and a price-change indicator), plus SNAP
and event-type features broken out per Phase 0's feature list.
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

LAG_DAYS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28]
PRICE_ROLL_WINDOW = 28

STATE_SNAP_COL = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}


def build_long_format(sales_sample: pd.DataFrame, day_cols: List[str],
                       calendar: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    sales_sample: a subset of rows from the wide sales table (already
    filtered to whatever series this run should use — see the sampling
    strategy in scripts/phase7_quantile_ml.py).
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    long_df = sales_sample.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")

    cal = calendar[["d", "date", "wm_yr_wk", "event_name_1", "event_type_1",
                     "snap_CA", "snap_TX", "snap_WI"]].copy()
    long_df = long_df.merge(cal, on="d", how="left")

    # Per-row SNAP flag for the row's OWN state -- fixes the hardcoded
    # snap_CA bug described in the module docstring.
    snap_matrix = long_df[["snap_CA", "snap_TX", "snap_WI"]].to_numpy()
    state_col_idx = long_df["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()
    long_df["snap"] = snap_matrix[np.arange(len(long_df)), state_col_idx]
    long_df = long_df.drop(columns=["snap_CA", "snap_TX", "snap_WI"])

    price_subset = prices.merge(sales_sample[["item_id", "store_id"]].drop_duplicates(),
                                 on=["item_id", "store_id"], how="inner")
    long_df = long_df.merge(price_subset, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    long_df["is_event"] = long_df["event_name_1"].notna().astype(int)
    long_df["sell_price"] = long_df.groupby("id")["sell_price"].ffill()

    return long_df.sort_values(["id", "date"]).reset_index(drop=True)


def add_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.sort_values(["id", "date"]).copy()
    g = df.groupby("id")["sales"]

    for lag in LAG_DAYS:
        if lag >= horizon:
            df[f"lag_{lag}"] = g.shift(lag)

    for w in ROLL_WINDOWS:
        shifted = g.shift(horizon)
        df[f"roll_mean_{w}"] = shifted.rolling(w).mean().reset_index(level=0, drop=True)
        df[f"roll_std_{w}"] = shifted.rolling(w).std().reset_index(level=0, drop=True)

    df["dow"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["item_code"] = df["item_id"].astype("category").cat.codes
    df["dept_code"] = df["dept_id"].astype("category").cat.codes
    df["cat_code"] = df["cat_id"].astype("category").cat.codes
    df["store_code"] = df["store_id"].astype("category").cat.codes

    # Price features. Treated as a KNOWN FUTURE covariate at the actual
    # forecast date, not lagged by `horizon` the way sales-derived
    # features are: a retailer sets its own prices and knows them in
    # advance, so using the price on the day being forecast is not
    # leakage the way using future SALES would be. This distinction is
    # deliberate, not an oversight -- see phase notes for the leakage
    # argument in full, and the test that checks it.
    price_by_id = df.groupby("id")["sell_price"]
    df["price_roll_mean"] = (price_by_id.transform(
        lambda s: s.rolling(PRICE_ROLL_WINDOW, min_periods=1).mean()))
    df["price_relative"] = df["sell_price"] / (df["price_roll_mean"] + 1e-8)
    df["price_change_flag"] = (price_by_id.diff().fillna(0) != 0).astype(int)

    return df


def feature_columns(horizon: int) -> List[str]:
    lags = [f"lag_{lag}" for lag in LAG_DAYS if lag >= horizon]
    rolls = [f"roll_mean_{w}" for w in ROLL_WINDOWS] + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    static = ["sell_price", "price_relative", "price_change_flag", "is_event", "snap",
              "dow", "day_of_year", "item_code", "dept_code", "cat_code", "store_code"]
    return lags + rolls + static
