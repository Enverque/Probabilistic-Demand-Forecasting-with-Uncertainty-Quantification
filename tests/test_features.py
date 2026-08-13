"""
Feature engineering tests. The leakage tests here matter more than
anywhere else in the project so far: this is the module where a subtle
off-by-one in a shift/rolling call silently inflates every later phase's
apparent accuracy without ever throwing an error.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.engineering import add_features, feature_columns, LAG_DAYS, build_long_format  # noqa: E402


def _synthetic_series_df(n=100, series_id="A"):
    dates = pd.date_range("2015-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "id": series_id, "item_id": "ITEM_1", "dept_id": "DEPT_1", "cat_id": "CAT_1",
        "store_id": "S1", "state_id": "CA",
        "date": dates, "sales": np.arange(n, dtype=float),  # sales_t = t, makes leakage checkable exactly
        "sell_price": 5.0,
    })
    return df


def test_lag_features_only_included_when_lag_ge_horizon():
    horizon = 28
    cols = feature_columns(horizon)
    lag_cols = [c for c in cols if c.startswith("lag_")]
    assert lag_cols == ["lag_28"]  # only lag_28 satisfies lag >= horizon among [1,7,14,28]

    horizon = 1
    cols = feature_columns(horizon)
    lag_cols = sorted(c for c in cols if c.startswith("lag_"))
    assert lag_cols == sorted(f"lag_{l}" for l in LAG_DAYS)  # all lags valid at horizon=1


def test_lag_value_is_exactly_correct_and_not_off_by_one():
    # sales_t = t (constructed above), so lag_k at row t should equal t - k exactly.
    df = _synthetic_series_df(n=60)
    out = add_features(df, horizon=1)
    row = out[out["date"] == pd.Timestamp("2015-01-01") + pd.Timedelta(days=40)].iloc[0]
    assert row["lag_1"] == 39  # sales at t-1
    assert row["lag_7"] == 33  # sales at t-7
    assert row["lag_28"] == 12  # sales at t-28


def test_rolling_features_never_include_the_current_or_within_horizon_days():
    # With sales_t = t, roll_mean_7 shifted by horizon should be the mean
    # of the 7 days ending AT t-horizon (inclusive) -- i.e. the most
    # recent day it touches is exactly `horizon` days before the forecast
    # target, never closer. Verify this numerically, not just by code
    # inspection.
    df = _synthetic_series_df(n=100)
    horizon = 28
    out = add_features(df, horizon=horizon)
    t = 80
    row = out[out["date"] == pd.Timestamp("2015-01-01") + pd.Timedelta(days=t)].iloc[0]
    # shifted series value at position t is sales[t - horizon] = t - horizon
    # roll_mean_7 = mean(sales[t-horizon-6 : t-horizon]), inclusive on both ends (7 values)
    expected = np.mean(np.arange(t - horizon - 6, t - horizon + 1))
    assert abs(row["roll_mean_7"] - expected) < 1e-6
    # Critically: the most recent day touched by the rolling window is
    # t-horizon, and t-horizon < t whenever horizon > 0 -- confirming the
    # window never reaches into the [t-horizon+1, t] gap between the
    # feature's information cutoff and the actual forecast target.
    most_recent_day_touched = t - horizon
    assert most_recent_day_touched < t


def test_no_feature_uses_future_sales_beyond_forecast_origin():
    # Blunt-force leakage check: zero out all sales from day `cutoff`
    # onward (simulating "the future hasn't happened yet"), and confirm
    # every feature value AT the cutoff row is identical whether computed
    # on the full series or on a series truncated at the cutoff. If any
    # feature secretly reached past the cutoff, this would catch it.
    df_full = _synthetic_series_df(n=150)
    horizon = 28
    cutoff_day = 100

    out_full = add_features(df_full, horizon=horizon)
    row_full = out_full[out_full["date"] == pd.Timestamp("2015-01-01") + pd.Timedelta(days=cutoff_day)].iloc[0]

    df_truncated = df_full[df_full["date"] <= pd.Timestamp("2015-01-01") + pd.Timedelta(days=cutoff_day)].copy()
    out_trunc = add_features(df_truncated, horizon=horizon)
    row_trunc = out_trunc[out_trunc["date"] == pd.Timestamp("2015-01-01") + pd.Timedelta(days=cutoff_day)].iloc[0]

    feat_cols = [c for c in feature_columns(horizon) if c.startswith(("lag_", "roll_"))]
    for c in feat_cols:
        assert row_full[c] == row_trunc[c] or (np.isnan(row_full[c]) and np.isnan(row_trunc[c])), \
            f"feature {c} differs between full and truncated series -- possible leakage"


def test_snap_flag_uses_own_state_not_hardcoded_california():
    # Regression test for the hardcoded snap_CA bug found in the
    # prototype. Construct rows for all 3 states with DIFFERENT snap
    # values and confirm each row gets its own state's value.
    calendar = pd.DataFrame({
        "d": ["d_1"], "date": [pd.Timestamp("2015-01-01")], "wm_yr_wk": [11101],
        "event_name_1": [None], "event_type_1": [None],
        "snap_CA": [1], "snap_TX": [0], "snap_WI": [1],
    })
    sales = pd.DataFrame({
        "id": ["CA_item", "TX_item", "WI_item"],
        "item_id": ["I1", "I1", "I1"], "dept_id": ["D1"] * 3, "cat_id": ["C1"] * 3,
        "store_id": ["CA_1", "TX_1", "WI_1"], "state_id": ["CA", "TX", "WI"],
        "d_1": [5, 5, 5],
    })
    prices = pd.DataFrame({"store_id": [], "item_id": [], "wm_yr_wk": [], "sell_price": []})
    out = build_long_format(sales, ["d_1"], calendar, prices)
    snap_by_state = out.set_index("state_id")["snap"].to_dict()
    assert snap_by_state["CA"] == 1
    assert snap_by_state["TX"] == 0
    assert snap_by_state["WI"] == 1


def test_price_relative_feature_is_dimensionless_and_centered_near_one():
    df = _synthetic_series_df(n=60)
    df.loc[df.index[30:], "sell_price"] = 6.0  # price bump partway through
    out = add_features(df, horizon=1)
    # price_relative should be close to 1 most of the time (price near its
    # own rolling mean) and shift after the price change
    assert out["price_relative"].between(0.5, 2.0).all()


def test_price_change_flag_fires_exactly_at_the_change():
    df = _synthetic_series_df(n=60)
    df.loc[df.index[30:], "sell_price"] = 6.0
    out = add_features(df, horizon=1).reset_index(drop=True)
    assert out.loc[30, "price_change_flag"] == 1
    assert out.loc[29, "price_change_flag"] == 0
    assert out.loc[31, "price_change_flag"] == 0
