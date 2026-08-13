"""
SARIMA for aggregated hierarchy levels only.

Not built for item-store granularity. Justification (see phase notes):
Phase 3 found 92%+ of item-store series classified lumpy/intermittent
(Syntetos-Boylan), with 68% zero-sales days (Phase 1) -- SARIMA's Gaussian-
innovation assumption is a poor fit for that regime, and fitting it to
30,490 series individually (Phase 0's explicit "computationally
inappropriate" warning) isn't justified when a better-suited model family
(quantile ML, Phase 7) is coming next anyway. SARIMA is fit here only on
aggregated series (Total, 10 Stores) where summing many underlying
intermittent series produces something much closer to continuous,
Gaussian-plausible demand -- itself worth confirming empirically, not
just assumed (see Phase 6 script).

Order selection: rather than a full auto-ARIMA search (pmdarima not used;
see phase notes for why), d and D are fixed from an ADF stationarity test
result (raw series non-stationary, first- and seasonal-differenced series
both stationary at p<0.001), and (p,q,P,Q) are chosen by a bounded AIC
grid search on the Total-level series only, then applied uniformly to all
Store-level series as a documented simplification -- not because every
store necessarily shares the same true order, but because a full
per-series order search across 11 series x 8 backtest windows would be
computationally disproportionate for this phase, and a shared order gives
a fair, if imperfect, apples-to-apples comparison to the other model
families in Phase 22.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings


def select_order_by_aic(series: np.ndarray, d: int, D: int, s: int,
                         p_range=(0, 1, 2), q_range=(0, 1, 2),
                         P_range=(0, 1), Q_range=(0, 1)) -> Dict:
    """Bounded grid search over (p,q,P,Q) with d,D,s fixed, ranked by AIC."""
    results = []
    for p in p_range:
        for q in q_range:
            for P in P_range:
                for Q in Q_range:
                    if p == 0 and q == 0 and P == 0 and Q == 0:
                        continue
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = SARIMAX(series, order=(p, d, q), seasonal_order=(P, D, Q, s),
                                             enforce_stationarity=False, enforce_invertibility=False)
                            fit = model.fit(disp=False)
                        results.append({"order": (p, d, q), "seasonal_order": (P, D, Q, s),
                                         "aic": fit.aic})
                    except Exception:
                        continue
    if not results:
        raise RuntimeError("No SARIMA configuration converged during order selection.")
    return min(results, key=lambda r: r["aic"])


def sarima_fit_predict(train: pd.DataFrame, test: pd.DataFrame,
                        order: Tuple[int, int, int], seasonal_order: Tuple[int, int, int, int],
                        target_col: str = "sales") -> pd.Series:
    """
    Fits fresh on `train` (no reuse of a previously-fit model across
    windows -- each backtest origin gets its own fit on exactly the data
    available at that origin) and forecasts len(test) steps ahead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SARIMAX(train[target_col].to_numpy(), order=order, seasonal_order=seasonal_order,
                         enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=len(test))
    forecast = np.clip(forecast, 0, None)  # sales can't be negative
    return pd.Series(forecast, index=test.index)
