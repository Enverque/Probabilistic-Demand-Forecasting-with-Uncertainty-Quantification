"""
Distributional properties of individual series: zero-inflation, heavy
tails, outliers, Croston-style intermittency classification, distribution
fitting.

Reused close to verbatim from `DistributionAnalyzer` in
`sales-forecast-probabilistic-eda-ml.ipynb` — this class was already
correct and well-scoped. Changes made:
- Removed the bare `except Exception: continue` in `fit_distributions`
  (silently swallowing all fit failures, including bugs, is a debugging
  liability) in favor of catching the specific exceptions scipy raises on
  bad fits.
- Added type hints cleanup; behavior is otherwise identical.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats


class DistributionAnalyzer:
    def __init__(self):
        self.distributions = ["norm", "lognorm", "gamma", "poisson", "nbinom", "expon", "weibull_min"]

    def analyze_zero_inflation(self, series: np.ndarray) -> Dict[str, float]:
        series = np.asarray(series, dtype=np.float64).flatten()
        n = len(series)
        zeros = np.sum(series == 0)
        zero_pct = zeros / n * 100
        mean_val = np.mean(series)
        expected_zero_pct = np.exp(-mean_val) * 100 if mean_val > 0 else 100.0
        zi_ratio = zero_pct / (expected_zero_pct + 1e-8)

        zero_mask = (series == 0).astype(int)
        zero_runs, current_run = [], 0
        for val in zero_mask:
            if val == 1:
                current_run += 1
            else:
                if current_run > 0:
                    zero_runs.append(current_run)
                current_run = 0
        if current_run > 0:
            zero_runs.append(current_run)

        return {
            "zero_count": int(zeros),
            "zero_pct": float(zero_pct),
            "expected_zero_pct": float(expected_zero_pct),
            "zi_ratio": float(zi_ratio),
            "is_zero_inflated": bool(zi_ratio > 1.5),
            "avg_zero_run": float(np.mean(zero_runs)) if zero_runs else 0.0,
            "max_zero_run": int(max(zero_runs)) if zero_runs else 0,
            "intermittency_index": float(zero_pct / 100),
        }

    def analyze_heavy_tails(self, series: np.ndarray) -> Dict[str, Any]:
        series = np.asarray(series, dtype=np.float64).flatten()
        series_nonzero = series[series > 0]
        if len(series_nonzero) < 4:
            return {"kurtosis": np.nan, "excess_kurtosis": np.nan, "skewness": np.nan,
                    "is_heavy_tailed": False, "tail_index": np.nan, "q99_q95_ratio": np.nan}

        kurt = stats.kurtosis(series_nonzero, fisher=True)
        skew = stats.skew(series_nonzero)
        sorted_series = np.sort(series_nonzero)[::-1]
        k = max(int(len(sorted_series) * 0.1), 5)
        if k < len(sorted_series):
            log_ratios = np.log(sorted_series[:k] / sorted_series[k])
            tail_index = 1 / np.mean(log_ratios) if np.mean(log_ratios) > 0 else np.nan
        else:
            tail_index = np.nan
        q95 = np.percentile(series_nonzero, 95)
        q99 = np.percentile(series_nonzero, 99)
        q99_q95_ratio = q99 / q95 if q95 > 0 else np.nan
        is_heavy_tailed = (kurt > 3) or (q99_q95_ratio > 2)

        return {
            "kurtosis": float(kurt + 3), "excess_kurtosis": float(kurt), "skewness": float(skew),
            "is_heavy_tailed": bool(is_heavy_tailed),
            "tail_index": float(tail_index) if not np.isnan(tail_index) else None,
            "q99_q95_ratio": float(q99_q95_ratio) if not np.isnan(q99_q95_ratio) else None,
        }

    def detect_outliers(self, series: np.ndarray, method: str = "iqr", threshold: float = 3.0) -> Dict[str, Any]:
        series = np.asarray(series, dtype=np.float64).flatten()
        if method == "iqr":
            q1, q3 = np.percentile(series, 25), np.percentile(series, 75)
            iqr = q3 - q1
            outliers = (series < q1 - threshold * iqr) | (series > q3 + threshold * iqr)
        elif method == "zscore":
            z = np.abs((series - np.mean(series)) / (np.std(series) + 1e-8))
            outliers = z > threshold
        elif method == "mad":
            median = np.median(series)
            mad = np.median(np.abs(series - median))
            modified_z = 0.6745 * (series - median) / (mad + 1e-8)
            outliers = np.abs(modified_z) > threshold
        else:
            raise ValueError(f"Unknown method: {method}")
        return {"method": method, "n_outliers": int(np.sum(outliers)),
                "outlier_pct": float(np.mean(outliers) * 100)}

    def fit_distributions(self, series: np.ndarray, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        NOTE ON A BUG FOUND HERE: the original prototype called
        `dist.logpdf(...)` for every distribution including 'poisson' and
        'nbinom', which are discrete (need `.logpmf`, not `.logpdf`), and
        called `dist.fit(...)` for 'nbinom', which doesn't exist in scipy
        for discrete distributions at all. Both errors were silently
        swallowed by a bare `except Exception: continue`, meaning the
        prototype's distribution fitting *always* silently dropped
        Poisson's likelihood computation and *always* silently excluded
        negative binomial entirely — on every series, with no error ever
        surfaced. That matters here specifically because NB is usually the
        best-motivated distribution for over-dispersed retail counts, so a
        pipeline that can never select it is a real gap, not a cosmetic
        one. Fixed below: discrete distributions use logpmf; NB is fit by
        method-of-moments (mean/variance -> n, p), since scipy provides no
        MLE `.fit()` for discrete distributions.
        """
        series = np.asarray(series, dtype=np.float64).flatten()
        series_positive = series[series > 0]
        if len(series_positive) < 10:
            return []
        results = []
        discrete = {"poisson", "nbinom"}
        for dist_name in self.distributions:
            try:
                dist = getattr(stats, dist_name)
                if dist_name == "poisson":
                    mu = np.mean(series_positive)
                    params = (mu,)
                    ks_stat, p_value = stats.kstest(series_positive, dist_name, args=params)
                    log_likelihood = np.sum(dist.logpmf(series_positive, *params))
                elif dist_name == "nbinom":
                    mean = np.mean(series_positive)
                    var = np.var(series_positive)
                    if var <= mean:
                        # Not over-dispersed relative to Poisson -> NB's
                        # method-of-moments solution is degenerate/invalid
                        # here; correctly skip rather than force a fit.
                        continue
                    p = mean / var
                    n = mean * p / (1 - p)
                    params = (n, p)
                    ks_stat, p_value = stats.kstest(series_positive, dist_name, args=params)
                    log_likelihood = np.sum(dist.logpmf(series_positive, *params))
                else:
                    params = dist.fit(series_positive)
                    ks_stat, p_value = stats.kstest(series_positive, dist_name, args=params)
                    log_likelihood = np.sum(dist.logpdf(series_positive, *params))
                k = len(params)
                aic = 2 * k - 2 * log_likelihood
                results.append({"distribution": dist_name, "aic": float(aic),
                                 "ks_statistic": float(ks_stat), "ks_pvalue": float(p_value)})
            except (ValueError, RuntimeError, FloatingPointError):
                continue
        return sorted(results, key=lambda x: x["aic"])[:top_n]

    def analyze_intermittency(self, series: np.ndarray) -> Dict[str, Any]:
        series = np.asarray(series, dtype=np.float64).flatten()
        nonzero_indices = np.where(series > 0)[0]
        if len(nonzero_indices) < 2:
            return {"intermittency_rate": 1.0, "avg_demand_interval": np.nan,
                     "avg_demand_size": np.nan, "classification": "dead"}
        intervals = np.diff(nonzero_indices)
        avg_interval = np.mean(intervals)
        avg_demand = np.mean(series[series > 0])
        intermittency_rate = np.sum(series == 0) / len(series)
        cv_demand = np.std(series[series > 0]) / (avg_demand + 1e-8)
        # Syntetos-Boylan classification
        if avg_interval < 1.32:
            classification = "erratic" if cv_demand >= 0.49 else "smooth"
        else:
            classification = "lumpy" if cv_demand >= 0.49 else "intermittent"
        return {"intermittency_rate": float(intermittency_rate), "avg_demand_interval": float(avg_interval),
                "avg_demand_size": float(avg_demand), "cv_demand": float(cv_demand),
                "classification": classification, "n_demand_occasions": int(len(nonzero_indices))}

    def analyze_series(self, series: np.ndarray, series_id: str = None) -> Dict[str, Any]:
        return {
            "series_id": series_id,
            "zero_inflation": self.analyze_zero_inflation(series),
            "heavy_tails": self.analyze_heavy_tails(series),
            "intermittency": self.analyze_intermittency(series),
            "outliers_iqr": self.detect_outliers(series, "iqr", 1.5),
            "best_distributions": self.fit_distributions(series, top_n=3),
        }

    def batch_analyze(self, series_dict: Dict[str, np.ndarray], verbose: bool = True) -> pd.DataFrame:
        rows = []
        for i, (series_id, series) in enumerate(series_dict.items()):
            if verbose and (i + 1) % 500 == 0:
                print(f"Analyzed {i + 1}/{len(series_dict)} series...")
            a = self.analyze_series(series, series_id)
            rows.append({
                "series_id": series_id,
                **{f"zi_{k}": v for k, v in a["zero_inflation"].items()},
                **{f"ht_{k}": v for k, v in a["heavy_tails"].items()},
                **{f"int_{k}": v for k, v in a["intermittency"].items()},
                "n_outliers_iqr": a["outliers_iqr"]["n_outliers"],
                "best_dist": a["best_distributions"][0]["distribution"] if a["best_distributions"] else None,
            })
        return pd.DataFrame(rows)
