"""
Explicit construction of the M5 hierarchy.

Refactored from `create_hierarchical_structure()` in
`sales-forecast-probabilistic-eda-ml.ipynb`. That prototype built 9 ad hoc
grouping levels (total, state, store, category, department, state_cat,
store_cat, store_dept, item). Checked against `weights_evaluation.csv`
(Phase 1) this misses 3 of the 12 levels the M5 evaluation actually
scores on: Level1 total is present but under a different construction,
Level7 (state x department) is missing entirely, Level10 (item, summed
across stores) is missing, and Level11 (item x state) is missing. All 12
are built here, named and ordered to match `Level1..Level12` in the
weights file so results can be joined back to the official weighting.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

# name -> list of groupby columns. [] means the single Total series.
# Order matches Level1..Level12 in weights_evaluation.csv.
HIERARCHY_LEVELS: Dict[str, List[str]] = {
    "Level1_total": [],
    "Level2_state": ["state_id"],
    "Level3_store": ["store_id"],
    "Level4_category": ["cat_id"],
    "Level5_department": ["dept_id"],
    "Level6_state_category": ["state_id", "cat_id"],
    "Level7_state_department": ["state_id", "dept_id"],
    "Level8_store_category": ["store_id", "cat_id"],
    "Level9_store_department": ["store_id", "dept_id"],
    "Level10_item": ["item_id"],
    "Level11_item_state": ["item_id", "state_id"],
    "Level12_item_store": ["item_id", "store_id"],
}


def build_hierarchy(sales: pd.DataFrame, day_cols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Aggregate the bottom-level (item x store) sales table up to each of
    the 12 official levels.

    Returns dict of level_name -> DataFrame with the groupby columns plus
    the day_cols summed. Level12 is returned as-is (it *is* the bottom
    level, aggregating it to itself would be a no-op that's easy to get
    subtly wrong, e.g. via unintended dedup) rather than re-derived.
    """
    out: Dict[str, pd.DataFrame] = {}
    for name, group_cols in HIERARCHY_LEVELS.items():
        if not group_cols:
            total = sales[day_cols].sum(axis=0).to_frame().T
            total.insert(0, "key", "Total")
            out[name] = total.reset_index(drop=True)
        elif group_cols == ["item_id", "store_id"]:
            out[name] = sales[["item_id", "store_id"] + day_cols].copy()
        else:
            out[name] = sales.groupby(group_cols)[day_cols].sum().reset_index()
    return out


def validate_against_weights(hierarchy: Dict[str, pd.DataFrame],
                              weights: pd.DataFrame) -> Dict[str, dict]:
    """
    Cross-check the number of series built at each level against the
    number of series the official weights file expects at that level —
    if the M5 organizers scored 30 store-category series and we built a
    different number, something about our grouping is wrong.
    """
    counts_expected = weights.groupby("Level_id").size().to_dict()
    results = {}
    for name in HIERARCHY_LEVELS:
        level_id = name.split("_")[0]  # "Level1_total" -> "Level1"
        expected = counts_expected.get(level_id)
        got = len(hierarchy[name])
        results[name] = {
            "passed": got == expected,
            "detail": f"got {got} series, weights file expects {expected}",
        }
    return results


def demonstrate_incoherence(sales: pd.DataFrame, day_cols: List[str],
                             naive_forecast_fn) -> pd.DataFrame:
    """
    Forecast Store level and State level *independently* using the same
    naive method applied to each level's own aggregated history, then
    check whether summing the store forecasts within a state matches the
    state's own independently-produced forecast. They generally won't —
    that gap is "incoherence," and is the empirical case for Phase 14.

    naive_forecast_fn: callable(np.ndarray of daily values) -> float,
    a one-step-ahead point forecast from a 1D history array.
    """
    store = sales.groupby("store_id")[day_cols].sum()
    state = sales.groupby("state_id")[day_cols].sum()
    store_to_state = sales[["store_id", "state_id"]].drop_duplicates().set_index("store_id")["state_id"]

    store_fc = store.apply(lambda row: naive_forecast_fn(row.to_numpy()), axis=1)
    state_fc_direct = state.apply(lambda row: naive_forecast_fn(row.to_numpy()), axis=1)

    store_fc_df = store_fc.to_frame("store_forecast")
    store_fc_df["state_id"] = store_to_state
    bottom_up_state_fc = store_fc_df.groupby("state_id")["store_forecast"].sum()

    comparison = pd.DataFrame({
        "state_forecast_direct": state_fc_direct,
        "state_forecast_bottom_up_from_stores": bottom_up_state_fc,
    })
    comparison["gap"] = comparison["state_forecast_direct"] - comparison["state_forecast_bottom_up_from_stores"]
    comparison["gap_pct"] = comparison["gap"] / comparison["state_forecast_direct"] * 100
    return comparison
