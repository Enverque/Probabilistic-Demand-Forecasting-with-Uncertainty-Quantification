"""
Fast, full-hierarchy-scale quantile forecaster for the dashboard.

The dashboard needs a forecast for an ARBITRARY user-selected slice of the
hierarchy (any state/store/category/department/item combination) computed
INTERACTIVELY -- retraining Phase 7's LightGBM quantile models per click
is not remotely fast enough for that (each fit takes minutes; Phases 7-13
compute exactly 3-8 backtest windows offline for that reason). Reuses the
weekday-quantile idea already proven correct and fast at full scale in
Phases 11/14 (seasonal_naive_wide), extended here to produce QUANTILES
(not just a point estimate), so the dashboard can show real uncertainty
bands for any selection without a training step.

This is a genuinely different, weaker model than Phase 7's LightGBM --
the dashboard says so explicitly (see app/dashboard.py's caveat text)
rather than implying it has the same accuracy the earlier phases measured.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


def quantile_seasonal_naive(series: np.ndarray, dow: np.ndarray, horizon: int,
                             quantiles: List[float], n_periods: int = 8) -> Dict[float, np.ndarray]:
    """
    series: 1D array of historical daily values (training period only).
    dow: day-of-week array aligned to `series` PLUS the horizon days ahead
        (length >= len(series) + horizon), so dow[len(series)+h] gives the
        weekday for forecast step h.
    Returns {quantile: array of length horizon}.
    """
    n = len(series)
    weekday_quantiles: Dict[int, Dict[float, float]] = {}
    for wd in range(7):
        idxs = np.flatnonzero(dow[:n] == wd)
        last_n = idxs[-n_periods:] if len(idxs) > 0 else idxs
        vals = series[last_n] if len(last_n) > 0 else series
        weekday_quantiles[wd] = {q: float(np.quantile(vals, q)) for q in quantiles}

    result = {q: np.zeros(horizon) for q in quantiles}
    for h in range(horizon):
        target_dow = int(dow[n + h])
        for q in quantiles:
            result[q][h] = weekday_quantiles[target_dow][q]

    # Enforce monotonicity across quantiles at each step -- np.quantile on
    # a small n_periods sample can occasionally produce near-ties that
    # aren't perfectly ordered after independent per-quantile computation.
    sorted_qs = sorted(quantiles)
    stacked = np.stack([result[q] for q in sorted_qs], axis=0)
    stacked = np.sort(stacked, axis=0)
    for i, q in enumerate(sorted_qs):
        result[q] = stacked[i]
    return result
