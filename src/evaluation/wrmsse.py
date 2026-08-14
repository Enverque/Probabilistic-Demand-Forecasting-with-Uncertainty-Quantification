"""
WRMSSE (Weighted Root Mean Squared Scaled Error), M5's official accuracy
metric. Implemented from the mathematical definition, not imported from a
library, per Phase 0's instruction to validate any such implementation
against a hand-computable example before trusting it (see
tests/test_wrmsse.py's hand-worked-example test).

Per series i, over an active training period of length n (starting at
that series' first non-zero observation -- a product's pre-launch zeros
shouldn't inflate its own scale) and a forecast horizon h:

    scale_i  = mean_{t=2..n} (Y_t - Y_{t-1})^2         [naive 1-step scale]
    RMSSE_i  = sqrt( mean_{t=n+1..n+h} (Y_t - Yhat_t)^2  /  scale_i )

Then, using the official per-series weights (revenue share, summing to 1
within each of the 12 aggregation levels):

    WRMSSE = (1/12) * sum_over_levels [ sum_i  w_i * RMSSE_i ]

i.e. each level contributes its own weighted-average RMSSE, and the 12
levels are then averaged equally -- this is why WRMSSE isn't just "RMSSE
averaged over all 42,840 series with global weights": levels with fewer,
more aggregated series (Level1 has exactly 1 series) get the same 1/12
share as Level12's 30,490 series.

The AGG_LEVEL_COLUMN_MAP below records the exact (Agg_Level_1, Agg_Level_2)
column order used by weights_evaluation.csv per level -- verified by
inspection, not assumed (Level11 is state-then-item, NOT item-then-state
the way Level12 is item-then-store; getting this backwards would silently
join every Level11 series to the wrong weight).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

AGG_LEVEL_COLUMN_MAP: Dict[str, List[str]] = {
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
    "Level11_item_state": ["state_id", "item_id"],   # note: state BEFORE item
    "Level12_item_store": ["item_id", "store_id"],
}


def scale_denominator(train_series: np.ndarray) -> Optional[float]:
    """
    Mean squared naive first-difference over the ACTIVE period only
    (from the series' first non-zero observation onward). A series that's
    never launched (all-zero) or launched with fewer than 2 active days
    has no defined scale -- returns None rather than 0 or NaN silently,
    forcing the caller to decide how to handle it explicitly.
    """
    nonzero_idx = np.flatnonzero(train_series)
    if len(nonzero_idx) == 0:
        return None
    active = train_series[nonzero_idx[0]:]
    if len(active) < 2:
        return None
    diffs = np.diff(active.astype(np.float64))
    denom = float(np.mean(diffs ** 2))
    return denom if denom > 0 else None


def rmsse(actual: np.ndarray, predicted: np.ndarray, train_series: np.ndarray) -> Optional[float]:
    denom = scale_denominator(train_series)
    if denom is None:
        return None
    mse = float(np.mean((actual.astype(np.float64) - predicted.astype(np.float64)) ** 2))
    return float(np.sqrt(mse / denom))


def wrmsse_for_level(rmsse_by_series: pd.Series, weights_by_series: pd.Series) -> float:
    """
    Weighted average RMSSE within one level. Series with an undefined
    RMSSE (None, from scale_denominator) are EXCLUDED and the remaining
    weights are NOT renormalized to sum to 1 among survivors -- excluding
    without renormalizing understates the level's score if many series
    are undefined, which is the conservative (not flattering) choice, and
    it's better than silently renormalizing over an arbitrary subset.
    """
    valid = rmsse_by_series.notna()
    if not valid.any():
        return float("nan")
    return float((rmsse_by_series[valid] * weights_by_series[valid]).sum())


def full_wrmsse(level_scores: Dict[str, float]) -> float:
    """Simple average of the 12 per-level weighted scores -- equal 1/12
    share per level regardless of how many series that level contains."""
    valid_scores = [v for v in level_scores.values() if not np.isnan(v)]
    if len(valid_scores) < len(level_scores):
        missing = [k for k, v in level_scores.items() if np.isnan(v)]
        raise ValueError(f"WRMSSE requires all 12 levels; missing/NaN: {missing}")
    return float(np.mean(valid_scores))
