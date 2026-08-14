"""
Pinball (quantile) loss, and Scaled Pinball Loss (SPL) -- the M5
Uncertainty-track analogue of WRMSSE, reusing the same per-series scale
denominator from wrmsse.py but applied to the pinball numerator instead of
squared error (and NOT square-rooted -- pinball loss is already in the
target's own units, unlike RMSSE which is a ratio of squared errors).

Built from the definition, validated by hand (tests/test_pinball.py)
before use, per Phase 0's instruction -- same discipline as Phase 11.

    pinball_tau(y, yhat) = tau * (y - yhat)          if y >= yhat
                          = (1 - tau) * (yhat - y)    if y <  yhat

Equivalently: max(tau*(y-yhat), (tau-1)*(y-yhat)) -- both forms are used
below (the piecewise form in pinball_loss for clarity/directness, the max
form implicitly checked to agree with it in tests).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.evaluation.wrmsse import scale_denominator


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    diff = y_true - y_pred
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return float(np.mean(loss))


def mean_pinball_across_quantiles(y_true: np.ndarray, preds_by_quantile: Dict[float, np.ndarray]) -> float:
    """Average pinball loss across several quantile levels -- the
    single-number summary used when comparing whole probabilistic
    forecasts (all quantiles at once), not just one quantile in isolation."""
    losses = [pinball_loss(y_true, preds_by_quantile[q], q) for q in preds_by_quantile]
    return float(np.mean(losses))


def scaled_pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float,
                         train_series: np.ndarray) -> Optional[float]:
    """
    NOTE ON THIS FORMULA: this divides pinball loss by sqrt(scale), where
    `scale` is the same mean-squared-naive-difference denominator WRMSSE
    uses -- a deliberate, unit-consistent choice (pinball loss is in
    absolute units; dividing by a squared-unit quantity without the sqrt
    would under-scale it), but I don't have a fully verified recall of
    M5's exact official SPL normalization in this offline environment
    (no web access to check against the competition's published formula).
    Treat this as a reasoned, documented normalization for this project
    rather than a guaranteed byte-for-byte match to the official metric --
    the same caution Phase 0 asks for anywhere a metric is used without
    being able to validate it against an authoritative source.
    """
    denom = scale_denominator(train_series)
    if denom is None:
        return None
    return pinball_loss(y_true, y_pred, tau) / np.sqrt(denom)
