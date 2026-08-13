"""
Trend, seasonality, holiday/price effects, and structural breaks.

None of this exists in either source notebook — notebook 1's EDA is
entirely distributional (per-series shape), and notebook 2 goes straight
to feature engineering without a seasonality/trend characterization step.
Built fresh here, deliberately kept to the smallest correct version per
concept rather than an exhaustive STL/Prophet-style decomposition, which
would be disproportionate for an EDA phase.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy import stats


def linear_trend(values: np.ndarray) -> Dict[str, float]:
    """
    OLS slope of value ~ day_index. Reports slope in units/day and its
    significance, not just direction — a 'trend' that isn't statistically
    distinguishable from noise shouldn't be reported as one.
    """
    x = np.arange(len(values))
    slope, intercept, r, p, se = stats.linregress(x, values)
    return {"slope_per_day": float(slope), "r_squared": float(r ** 2), "p_value": float(p)}


def weekly_seasonality(values: np.ndarray, dow: np.ndarray) -> Dict[str, float]:
    """
    One-way ANOVA of value ~ day_of_week. F-statistic and p-value tell us
    whether day-of-week explains a significant share of variance; the
    per-weekday means show the *shape* of the effect.
    """
    groups = [values[dow == d] for d in range(7)]
    f_stat, p_value = stats.f_oneway(*groups)
    means = {int(d): float(np.mean(values[dow == d])) for d in range(7)}
    peak_day = max(means, key=means.get)
    trough_day = min(means, key=means.get)
    return {"f_statistic": float(f_stat), "p_value": float(p_value),
            "means_by_dow": means, "peak_dow": peak_day, "trough_dow": trough_day,
            "peak_to_trough_ratio": means[peak_day] / (means[trough_day] + 1e-8)}


def monthly_seasonality(values: np.ndarray, month: np.ndarray) -> Dict[str, float]:
    """Same idea as weekly_seasonality but for month-of-year (proxy for yearly seasonality)."""
    groups = [values[month == m] for m in range(1, 13) if (month == m).any()]
    f_stat, p_value = stats.f_oneway(*groups) if len(groups) > 1 else (np.nan, np.nan)
    means = {int(m): float(np.mean(values[month == m])) for m in range(1, 13) if (month == m).any()}
    return {"f_statistic": float(f_stat), "p_value": float(p_value), "means_by_month": means}


def holiday_event_effect(values: np.ndarray, has_event: np.ndarray) -> Dict[str, float]:
    """
    Welch's t-test comparing mean sales on event days vs. non-event days.
    Welch's (not Student's) t-test because event days are a small, likely
    unequal-variance minority of the sample — assuming equal variance here
    would be the wrong default.
    """
    event_vals = values[has_event == 1]
    non_event_vals = values[has_event == 0]
    if len(event_vals) < 2:
        return {"event_mean": np.nan, "non_event_mean": np.nan, "p_value": np.nan}
    t_stat, p_value = stats.ttest_ind(event_vals, non_event_vals, equal_var=False)
    return {
        "event_mean": float(event_vals.mean()), "non_event_mean": float(non_event_vals.mean()),
        "pct_lift": float((event_vals.mean() - non_event_vals.mean()) / (non_event_vals.mean() + 1e-8) * 100),
        "t_statistic": float(t_stat), "p_value": float(p_value),
    }


def price_sales_correlation(prices: np.ndarray, sales: np.ndarray) -> Dict[str, float]:
    """
    Spearman correlation between price and sales for a single series.
    Spearman, not Pearson, because the price-demand relationship is
    expected to be monotonic but not necessarily linear, and because
    sales data has many ties (zeros) that distort Pearson's assumptions.
    Deliberately NOT labeled a causal elasticity estimate — see Phase 17.
    """
    mask = ~np.isnan(prices)
    if mask.sum() < 10 or np.unique(prices[mask]).size < 2:
        return {"spearman_r": np.nan, "p_value": np.nan, "n_price_changes": 0}
    r, p = stats.spearmanr(prices[mask], sales[mask])
    n_changes = int(np.sum(np.diff(prices[mask]) != 0))
    return {"spearman_r": float(r), "p_value": float(p), "n_price_changes": n_changes}


def structural_break_scan(values: np.ndarray, window: int = 90, min_gap: int = 180) -> Dict[str, object]:
    """
    Lightweight structural-break scan: compare the mean of each `window`-day
    block to the immediately preceding block of equal length, flag the
    largest jump (in units of the preceding block's standard deviation) as
    a candidate break. This is deliberately simple (not CUSUM/Bai-Perron) —
    Phase 0 already said not to add a sophisticated change-point method
    unless it earns its place; this is enough to locate *candidates* for
    visual inspection, not to make a statistical claim of exactly one
    change point.
    """
    n = len(values)
    if n < 2 * window:
        return {"n_candidates": 0, "top_break_day": None, "top_break_z": None}
    scores = []
    for start in range(window, n - window, window // 3):
        prev_block = values[start - window:start]
        next_block = values[start:start + window]
        prev_std = prev_block.std() + 1e-8
        z = (next_block.mean() - prev_block.mean()) / prev_std
        scores.append((start, z))
    if not scores:
        return {"n_candidates": 0, "top_break_day": None, "top_break_z": None}
    top = max(scores, key=lambda t: abs(t[1]))
    n_large = sum(1 for _, z in scores if abs(z) > 2.0)
    return {"n_candidates": n_large, "top_break_day": int(top[0]), "top_break_z": float(top[1])}
