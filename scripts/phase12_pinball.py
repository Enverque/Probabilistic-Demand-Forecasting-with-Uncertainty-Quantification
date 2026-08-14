"""
Phase 12: Probabilistic Metrics (Pinball Loss).

Applies pinball loss to the SAME quantile predictions generated in Phase 8
(same 600-series sample, same train/test split, full P2.5..P97.5 set) --
this phase evaluates output that already exists, it doesn't regenerate it
differently. Compares against a naive EMPIRICAL-QUANTILE baseline (predict
each series' own training-period quantile as a constant) so the pinball
numbers have a reference point, not just a standalone value -- the same
"baselines before judging the ML model" discipline as every other phase.

Run: python scripts/phase12_pinball.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.features.engineering import build_long_format, add_features, feature_columns  # noqa: E402
from src.forecasting.probabilistic import generate_probabilistic_forecast, QUANTILE_LEVELS  # noqa: E402
from src.evaluation.pinball import pinball_loss, mean_pinball_across_quantiles, scaled_pinball_loss  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 200
HORIZON = 28
TRAIN_END = pd.Timestamp("2016-04-24")
TEST_END = pd.Timestamp("2016-05-22")


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))
    long_df = build_long_format(sample, data.day_cols, data.calendar, data.prices)
    featured = add_features(long_df, horizon=HORIZON)
    numeric_cols = featured.select_dtypes(include=["float64"]).columns
    featured[numeric_cols] = featured[numeric_cols].astype("float32")
    cols = feature_columns(HORIZON)

    train = featured[featured["date"] <= TRAIN_END].dropna(subset=cols)
    test = featured[(featured["date"] > TRAIN_END) & (featured["date"] <= TEST_END)].dropna(subset=cols)
    print(f"Train: {len(train)} rows  |  Test: {len(test)} rows")

    print(f"\nGenerating quantile ML predictions ({QUANTILE_LEVELS})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ml_preds = generate_probabilistic_forecast(train, test, cols, "sales", random_state=RANDOM_STATE)
    test = test.reset_index(drop=True)
    ml_preds = ml_preds.reset_index(drop=True)

    print("Generating empirical-quantile baseline (each series' own training-period quantile)...")
    empirical_q = train.groupby("id")["sales"].quantile(QUANTILE_LEVELS).unstack()
    baseline_preds = pd.DataFrame(index=test.index, columns=sorted(QUANTILE_LEVELS), dtype=float)
    for q in QUANTILE_LEVELS:
        baseline_preds[q] = test["id"].map(empirical_q[q]).to_numpy()

    print("\n--- Pinball loss per quantile ---")
    print(f"{'quantile':>10} {'ML':>12} {'empirical-Q baseline':>22}")
    ml_losses, baseline_losses = {}, {}
    for q in sorted(QUANTILE_LEVELS):
        l_ml = pinball_loss(test["sales"], ml_preds[q], q)
        l_base = pinball_loss(test["sales"], baseline_preds[q], q)
        ml_losses[q] = l_ml
        baseline_losses[q] = l_base
        print(f"{q:>10.3f} {l_ml:>12.4f} {l_base:>22.4f}")

    mean_ml = mean_pinball_across_quantiles(
        test["sales"].to_numpy(), {q: ml_preds[q].to_numpy() for q in QUANTILE_LEVELS})
    mean_base = mean_pinball_across_quantiles(
        test["sales"].to_numpy(), {q: baseline_preds[q].to_numpy() for q in QUANTILE_LEVELS})
    print("\nMean pinball loss across all 7 quantiles:")
    print(f"  Quantile ML:                 {mean_ml:.4f}")
    print(f"  Empirical-quantile baseline: {mean_base:.4f}")
    print(f"  Relative improvement: {(mean_base - mean_ml) / mean_base * 100:+.1f}%")

    print("\n--- Scaled pinball loss (P50), a sample of 10 series, for illustration ---")
    sample_ids = test["id"].unique()[:10]
    for sid in sample_ids:
        mask = (test["id"] == sid).to_numpy()
        train_arr = train.loc[train["id"] == sid, "sales"].to_numpy()
        spl_ml = scaled_pinball_loss(test.loc[mask, "sales"].to_numpy(), ml_preds.loc[mask, 0.5].to_numpy(),
                                      0.5, train_arr)
        label = f"{spl_ml:.3f}" if spl_ml is not None else "undefined (flat/zero training scale)"
        print(f"  {sid}: SPL(P50) = {label}")


if __name__ == "__main__":
    main()
