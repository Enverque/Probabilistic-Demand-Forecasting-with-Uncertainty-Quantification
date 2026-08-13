"""
Phase 9: Deep Probabilistic Model (optional) -- Bayesian structural time
series, the real candidate available (notebook 1), evaluated honestly
rather than defaulted into.

Runs ONE real fit + forecast on the Total-level series (single train/test
split, matching the same 2016-04-24 cutoff used in Phase 8, for
comparability), with reduced-but-documented sampling settings, then states
the feasibility conclusion for whether this belongs in the project's
final model set (Phase 22) given real timing evidence, not assumption.

Run: python scripts/phase9_bayesian.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy  # noqa: E402
from src.models.bayesian_structural import BayesianStructuralModel  # noqa: E402
from src.evaluation.metrics import wmape, bias  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
TRAIN_END = pd.Timestamp("2016-04-24")
TEST_END = pd.Timestamp("2016-05-22")


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    hierarchy = build_hierarchy(data.sales, data.day_cols)
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)
    total_vals = hierarchy["Level1_total"].iloc[0][data.day_cols].to_numpy(dtype=float)

    df = pd.DataFrame({"date": cal["date"], "sales": total_vals})
    train = df[df["date"] <= TRAIN_END]
    test = df[(df["date"] > TRAIN_END) & (df["date"] <= TEST_END)]
    print(f"Train: {len(train)} days up to {TRAIN_END.date()}  |  Test: {len(test)} days")

    print("\nBuilding and fitting Bayesian structural model "
          "(trend + weekly + monthly seasonality, Negative Binomial likelihood)...")
    print("Settings: draws=500, tune=500, chains=2, cores=1 "
          "(reduced from prototype default draws=2000/tune=1000/chains=4 -- "
          "see src/models/bayesian_structural.py docstring for the timing justification)")

    t0 = time.time()
    model = BayesianStructuralModel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.build_model(train["sales"].to_numpy(), use_weekly=True, use_monthly=True)
        trace = model.fit(draws=500, tune=500, chains=2, target_accept=0.9)
    fit_time = time.time() - t0
    print(f"Fit completed in {fit_time:.0f}s ({fit_time/60:.1f} minutes)")

    print("\n--- Convergence diagnostics ---")
    summary = az.summary(trace, var_names=["sigma_trend", "alpha"])
    print(summary[["mean", "sd", "r_hat", "ess_bulk"]].to_string())
    max_rhat = az.summary(trace)["r_hat"].max()
    print(f"Max r_hat across ALL parameters: {max_rhat:.3f} "
          f"({'OK, <1.01' if max_rhat < 1.01 else 'CONVERGENCE CONCERN, >1.01'})")

    print(f"\nForecasting {len(test)} days ahead...")
    t0 = time.time()
    forecast = model.forecast(steps=len(test), num_samples=300)
    print(f"Forecast sampling done in {time.time()-t0:.0f}s")

    actual = test["sales"].to_numpy()
    point_wmape = wmape(actual, forecast["forecast_median"])
    point_bias = bias(actual, forecast["forecast_median"])
    cov_50 = ((actual >= forecast["lower_50"]) & (actual <= forecast["upper_50"])).mean()
    cov_95 = ((actual >= forecast["lower_95"]) & (actual <= forecast["upper_95"])).mean()

    print("\n--- Forecast results (single series, single window) ---")
    print(f"Median-forecast WMAPE: {point_wmape*100:.1f}%  (Phase 6 SARIMA on same series: 6.6%, "
          f"Phase 5 best naive: 6.1%)")
    print(f"Bias: {point_bias*100:+.1f}%")
    print(f"50% interval empirical coverage: {cov_50*100:.1f}% (nominal 50%)")
    print(f"95% interval empirical coverage: {cov_95*100:.1f}% (nominal 95%)")

    print("\n--- Feasibility conclusion (Phase 0 explicitly requires this before adopting the model) ---")
    single_series_time_min = fit_time / 60
    n_windows_phase4 = 8
    n_series_hierarchy = 30490
    print(f"Single fit took {single_series_time_min:.1f} minutes on 1 CPU core.")
    print(f"A full Phase 4-style backtest (8 windows) on just this ONE series would take "
          f"~{single_series_time_min*8:.0f} minutes.")
    print(f"Running this at item-store granularity ({n_series_hierarchy:,} series) is not "
          f"remotely tractable in this environment: "
          f"~{single_series_time_min*n_series_hierarchy/60/24:.0f} CPU-days for a single window alone.")


if __name__ == "__main__":
    main()
