"""
Phase 21: Experiment Tracking.

Consolidates the REAL numbers already reported in Phases 5-16's own script
output into a single log. Every value below was transcribed from that
phase's actual printed results (cited in the `notes` field), not
recomputed or invented -- Phase 0's "do not invent numbers" rule applies
to this retrospective transcription exactly as it did to the original
measurement.

Run: python scripts/phase21_experiment_log.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.experiment_log import log_experiment, get_experiment_log  # noqa: E402

LOG_PATH = str(Path(__file__).resolve().parents[1] / "data" / "interim" / "experiment_log.csv")

RECORDS = [
    dict(phase="Phase5", model="naive_last_value", features="none",
         forecast_horizon=28, backtest_windows=8, series_scope="2100-series sample (700/cat)",
         metric_name="WMAPE", metric_value=1.059,
         runtime_notes="fast, no training", notes="mean of 8 windows, item-store level"),
    dict(phase="Phase5", model="seasonal_naive", features="none",
         forecast_horizon=28, backtest_windows=8, series_scope="2100-series sample (700/cat)",
         metric_name="WMAPE", metric_value=0.856,
         runtime_notes="fast, no training", notes="mean of 8 windows, item-store level"),
    dict(phase="Phase5", model="moving_average_7d", features="none",
         forecast_horizon=28, backtest_windows=8, series_scope="2100-series sample (700/cat)",
         metric_name="WMAPE", metric_value=0.838,
         runtime_notes="fast, no training", notes="mean of 8 windows, item-store level"),

    dict(phase="Phase6", model="SARIMA(0,1,2)(1,1,1,7)", features="none (own history only)",
         forecast_horizon=28, backtest_windows=4, series_scope="Total level (1 series)",
         metric_name="WMAPE", metric_value=0.066,
         runtime_notes="~21s order selection + backtest fits", notes="best naive on same series: 6.1% WMAPE"),
    dict(phase="Phase6", model="SARIMA_vs_naive_mean_FVA", features="none",
         forecast_horizon=28, backtest_windows=4, series_scope="Total + 10 stores (11 series)",
         metric_name="mean_FVA_vs_best_naive", metric_value=-0.027,
         runtime_notes="~30 min total for 11 series", notes="SARIMA beat naive in only 4/11 series"),

    dict(phase="Phase7", model="quantile_lightgbm_P50", features="lag_28,roll_mean/std_7/28,price,calendar,snap",
         forecast_horizon=28, backtest_windows=3, series_scope="600-series sample (200/cat)",
         metric_name="WMAPE", metric_value=0.738,
         runtime_notes="~7 min for 3 windows", notes="best naive on same sample: 83.1% (seasonal_naive)"),
    dict(phase="Phase7", model="quantile_lightgbm_P50", features="same as above",
         forecast_horizon=28, backtest_windows=3, series_scope="600-series sample (200/cat)",
         metric_name="FVA_vs_best_naive", metric_value=0.086,
         runtime_notes="~7 min for 3 windows", notes=""),
    dict(phase="Phase7", model="quantile_lightgbm_P10_P90", features="same as above",
         forecast_horizon=28, backtest_windows=3, series_scope="600-series sample (200/cat)",
         metric_name="coverage_80pct_interval", metric_value=0.918,
         runtime_notes="~7 min for 3 windows", notes="nominal target 80%, overcovered"),

    dict(phase="Phase9", model="Bayesian_structural_NB", features="trend+weekly+monthly Fourier",
         forecast_horizon=28, backtest_windows=1, series_scope="Total level (1 series)",
         metric_name="WMAPE", metric_value=0.101,
         runtime_notes="735s (12.2 min) for 1 series, 1 window, 1 CPU core",
         notes="NOT RELIABLE: max r_hat=3.04, 41 divergences, non-converged posterior. Not adopted."),

    dict(phase="Phase10", model="global_lightgbm_point", features="same feature set as Phase 7",
         forecast_horizon=28, backtest_windows=1, series_scope="550 normal-history series",
         metric_name="WMAPE", metric_value=0.800,
         runtime_notes="single window", notes="post rolling-leakage-bug fix"),
    dict(phase="Phase10", model="local_lightgbm_point (one model per series)", features="same feature set",
         forecast_horizon=28, backtest_windows=1, series_scope="550 normal-history series",
         metric_name="WMAPE", metric_value=0.7894,
         runtime_notes="single window, 550 separate fits", notes="post rolling-leakage-bug fix"),
    dict(phase="Phase10", model="global_lightgbm_point", features="same feature set",
         forecast_horizon=28, backtest_windows=1, series_scope="50 cold-start (60-day truncated history)",
         metric_name="WMAPE", metric_value=0.6110,
         runtime_notes="single window", notes="50/50 local models fell back to constant on this subset"),
    dict(phase="Phase10", model="local_lightgbm_point (fallback-heavy)", features="same feature set",
         forecast_horizon=28, backtest_windows=1, series_scope="50 cold-start (60-day truncated history)",
         metric_name="WMAPE", metric_value=0.6304,
         runtime_notes="single window", notes="50/50 fell back to a constant, too little data to fit"),

    dict(phase="Phase11", model="seasonal_naive", features="none",
         forecast_horizon=28, backtest_windows=1, series_scope="ALL 30,490 series, all 12 levels",
         metric_name="full_official_WRMSSE", metric_value=0.7695,
         runtime_notes="vectorized wide-array, ~2 min", notes="official weights, all levels"),
    dict(phase="Phase11", model="quantile_lightgbm_P50", features="same as Phase 7",
         forecast_horizon=28, backtest_windows=1, series_scope="600-series sample, Level12 only",
         metric_name="sample_scope_WRMSSE_Level12", metric_value=0.9534,
         runtime_notes="single window", notes="renormalized weights, NOT directly comparable to full-scale number"),

    dict(phase="Phase12", model="quantile_lightgbm (7 quantiles)", features="same as Phase 7",
         forecast_horizon=28, backtest_windows=1, series_scope="600-series sample",
         metric_name="mean_pinball_loss_7_quantiles", metric_value=0.2693,
         runtime_notes="single window, 7 model fits", notes="empirical-quantile baseline: 0.2886 (+6.7% worse)"),

    dict(phase="Phase13", model="quantile_lightgbm", features="same as Phase 7",
         forecast_horizon=28, backtest_windows=3, series_scope="600-series sample, pooled",
         metric_name="coverage_50pct_interval", metric_value=0.743,
         runtime_notes="3 windows, 7 quantile fits each", notes="nominal 50%, overcovered by 24.3pp"),
    dict(phase="Phase13", model="quantile_lightgbm", features="same as Phase 7",
         forecast_horizon=28, backtest_windows=3, series_scope="600-series sample, pooled",
         metric_name="coverage_95pct_interval", metric_value=0.975,
         runtime_notes="3 windows, 7 quantile fits each", notes="nominal 95%, overcovered by 2.5pp only"),

    dict(phase="Phase14", model="seasonal_naive_median_DIRECT", features="none",
         forecast_horizon=28, backtest_windows=1, series_scope="ALL 30,490 series, all 12 levels",
         metric_name="full_WRMSSE", metric_value=0.7795,
         runtime_notes="vectorized, ~2 min", notes="median-based to avoid mean's linear-coherence degenerate case"),
    dict(phase="Phase14", model="seasonal_naive_median_BOTTOMUP_reconciled", features="none",
         forecast_horizon=28, backtest_windows=1, series_scope="ALL 30,490 series, all 12 levels",
         metric_name="full_WRMSSE", metric_value=1.2328,
         runtime_notes="vectorized, ~2 min", notes="58% worse than DIRECT; coherent by construction (gap=0 exactly)"),

    dict(phase="Phase16", model="point_policy(P50)_vs_probabilistic", features="Phase 8 quantile outputs",
         forecast_horizon=28, backtest_windows=1, series_scope="600-series sample",
         metric_name="cost_reduction_stockout_expensive_4to1", metric_value=0.265,
         runtime_notes="single window", notes="critical ratio 0.80"),
    dict(phase="Phase16", model="point_policy(P50)_vs_probabilistic", features="Phase 8 quantile outputs",
         forecast_horizon=28, backtest_windows=1, series_scope="600-series sample",
         metric_name="cost_reduction_holding_expensive_1to3", metric_value=0.106,
         runtime_notes="single window", notes="critical ratio 0.25"),
]


def main():
    if Path(LOG_PATH).exists():
        Path(LOG_PATH).unlink()
    for r in RECORDS:
        log_experiment(r, LOG_PATH)

    log = get_experiment_log(LOG_PATH)
    print(f"Experiment log written: {LOG_PATH}")
    print(f"{len(log)} records from {log['phase'].nunique()} phases\n")
    print(log[["phase", "model", "series_scope", "metric_name", "metric_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
