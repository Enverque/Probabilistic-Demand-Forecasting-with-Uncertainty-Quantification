"""
Point-forecast metrics. wmape/bias/fva reused verbatim from notebook 2 —
correct as written, no changes needed. This module will grow in Phase 11
(WRMSSE) and Phase 12 (pinball loss); kept minimal here rather than
pre-building metrics this phase doesn't need yet.
"""
import numpy as np
import pandas as pd


def wmape(y, p) -> float:
    """Weighted MAPE: sum|error| / sum|actual|. Well-defined even with
    zero-sales days (unlike ordinary MAPE, which divides by y itself)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / (np.abs(y).sum() + 1e-8))


def bias(y, p) -> float:
    """Signed relative error: positive = over-forecasting, negative = under."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float((p.sum() - y.sum()) / (y.sum() + 1e-8))


def fva(baseline_metric: float, model_metric: float) -> float:
    """Forecast Value Added: relative improvement of model over baseline
    on the same (lower-is-better) metric."""
    return float((baseline_metric - model_metric) / (baseline_metric + 1e-8))
