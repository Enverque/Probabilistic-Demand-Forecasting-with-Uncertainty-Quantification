"""
Formalizes Phase 7's quantile predictions into named prediction intervals.

Phase 7 already built the mechanics (fit_quantiles/predict_quantiles,
monotonicity-enforced); this module doesn't duplicate that -- it wraps it
with the full quantile set Phase 0 specified (P2.5/P10/P25/P50/P75/P90/
P97.5) and turns raw quantile columns into explicit, named intervals
(50%/80%/95%) that Phase 13 (calibration) and Phase 16 (inventory
simulation) can consume directly instead of re-deriving interval bounds
from quantile column names themselves.
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.models.quantile_ml import fit_quantiles, predict_quantiles

QUANTILE_LEVELS: List[float] = [0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]

# name -> (lower_quantile, upper_quantile). Central intervals only --
# each pair is symmetric around P50 in probability mass, not necessarily
# in value (demand distributions are skewed, so a 50% interval will
# generally NOT be symmetric in units around the median -- see Phase 8
# script for a real example of this).
INTERVAL_DEFINITIONS: Dict[str, tuple] = {
    "50%": (0.25, 0.75),
    "80%": (0.10, 0.90),
    "95%": (0.025, 0.975),
}


def generate_probabilistic_forecast(train: pd.DataFrame, test: pd.DataFrame,
                                     feature_cols: List[str], target_col: str = "sales",
                                     quantiles: List[float] = None,
                                     random_state: int = 42) -> pd.DataFrame:
    """Fits all quantiles in `quantiles` (default: the full Phase 0 set) and
    returns a DataFrame of monotonic quantile predictions, one column per
    quantile level, indexed like `test`."""
    quantiles = quantiles or QUANTILE_LEVELS
    models = fit_quantiles(train, feature_cols, target_col, quantiles, random_state=random_state)
    return predict_quantiles(models, test, feature_cols)


def build_intervals(quantile_preds: pd.DataFrame,
                     interval_defs: Dict[str, tuple] = None) -> pd.DataFrame:
    """
    Turns quantile columns into named interval columns:
    {name}_lower, {name}_upper, {name}_width, plus the median as 'point'.

    Raises if a requested interval's quantile columns weren't actually
    predicted -- silently returning NaN here would be a much worse
    failure mode than an explicit error, since a downstream inventory
    decision (Phase 16) consuming a silently-NaN interval could produce a
    nonsensical order quantity without any visible error.
    """
    interval_defs = interval_defs or INTERVAL_DEFINITIONS
    out = pd.DataFrame(index=quantile_preds.index)
    if 0.5 in quantile_preds.columns:
        out["point"] = quantile_preds[0.5]
    for name, (lo_q, hi_q) in interval_defs.items():
        if lo_q not in quantile_preds.columns or hi_q not in quantile_preds.columns:
            raise KeyError(f"Interval '{name}' requires quantiles {lo_q} and {hi_q}, "
                            f"but only {list(quantile_preds.columns)} were predicted.")
        out[f"{name}_lower"] = quantile_preds[lo_q]
        out[f"{name}_upper"] = quantile_preds[hi_q]
        out[f"{name}_width"] = quantile_preds[hi_q] - quantile_preds[lo_q]
    return out


def skewness_of_intervals(intervals: pd.DataFrame, interval_name: str = "80%") -> pd.Series:
    """
    How far the median sits from the interval's midpoint, as a fraction of
    the interval's half-width. 0 = median exactly centered (symmetric
    interval); positive = median closer to the lower bound (right-skewed,
    the expected direction for demand data per Phase 3's heavy-tail
    findings); negative = median closer to the upper bound.
    """
    lower = intervals[f"{interval_name}_lower"]
    upper = intervals[f"{interval_name}_upper"]
    midpoint = (lower + upper) / 2
    half_width = (upper - lower) / 2
    return (midpoint - intervals["point"]) / (half_width + 1e-8)
