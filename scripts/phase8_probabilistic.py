"""
Phase 8: Probabilistic Forecasting.

Generates the full P2.5/P10/P25/P50/P75/P90/P97.5 quantile set and
50%/80%/95% intervals on the real 600-series sample from Phase 7, using
the SAME features/data -- this phase formalizes Phase 7's output, it
doesn't re-do the modeling. Trains once on the most recent backtest
window's train/test split (not a full multi-window backtest -- that's
what Phase 7 already did for the point/median comparison, and what
Phase 13 will do for calibration across all windows).

Shows the result concretely for two contrasting real series: the
highest-volume series in the sample (demand shape closer to continuous)
and a genuinely intermittent one (many zero days) -- because "here's a
prediction interval" means something different for each.

Run: python scripts/phase8_probabilistic.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.features.engineering import build_long_format, add_features, feature_columns  # noqa: E402
from src.forecasting.probabilistic import (  # noqa: E402
    generate_probabilistic_forecast, build_intervals, skewness_of_intervals, QUANTILE_LEVELS,
)

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
    print(f"Train: {len(train)} rows up to {TRAIN_END.date()}  |  Test: {len(test)} rows")

    print(f"\nFitting {len(QUANTILE_LEVELS)} quantile models: {QUANTILE_LEVELS} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds = generate_probabilistic_forecast(train, test, cols, "sales", random_state=RANDOM_STATE)
    intervals = build_intervals(preds)
    test = test.reset_index(drop=True)
    intervals = intervals.reset_index(drop=True)
    preds = preds.reset_index(drop=True)

    print("\nQuantile definitions:")
    for q in sorted(QUANTILE_LEVELS):
        print(f"  P{q*100:.1f}: demand value such that ~{q*100:.1f}% of outcomes are expected "
              f"to fall below it, subject to model calibration (verified empirically, not "
              f"assumed -- see Phase 13).")

    test_totals = test.groupby("id")["sales"].sum()
    high_volume_id = test_totals.idxmax()
    train_zero_rate = train.groupby("id")["sales"].apply(lambda s: (s == 0).mean())
    intermittent_candidates = train_zero_rate[(train_zero_rate > 0.7) & (train_zero_rate < 0.95)]
    intermittent_id = intermittent_candidates.index[0] if len(intermittent_candidates) else None

    for label, series_id in [("HIGH-VOLUME", high_volume_id), ("INTERMITTENT", intermittent_id)]:
        if series_id is None:
            continue
        mask = (test["id"] == series_id).to_numpy()
        print(f"\n--- {label} series: {series_id} ---")
        sub_actual = test.loc[mask, "sales"].reset_index(drop=True)
        sub_preds = preds.loc[mask].reset_index(drop=True)
        sub_intervals = intervals.loc[mask].reset_index(drop=True)
        print(f"Actual test-period sales: mean={sub_actual.mean():.2f}, "
              f"zero-rate={100*(sub_actual==0).mean():.1f}%")
        print("First 5 forecast days:")
        display_cols = sorted(QUANTILE_LEVELS)
        print(pd.concat([sub_preds[display_cols].head(5).round(2),
                          sub_actual.head(5).rename("actual")], axis=1).to_string(index=False))
        print(f"Mean 50% interval width: {sub_intervals['50%_width'].mean():.2f}")
        print(f"Mean 80% interval width: {sub_intervals['80%_width'].mean():.2f}")
        print(f"Mean 95% interval width: {sub_intervals['95%_width'].mean():.2f}")
        skew = skewness_of_intervals(sub_intervals, "80%")
        print(f"Mean interval skewness (80%, +=right-skewed toward lower bound): {skew.mean():+.3f}")
        covered_80 = ((sub_actual >= sub_intervals["80%_lower"]) & (sub_actual <= sub_intervals["80%_upper"])).mean()
        print(f"Empirical 80% coverage on this single series: {covered_80*100:.1f}% "
              f"(n={len(sub_actual)} days -- too few to judge calibration from alone, see Phase 13)")


if __name__ == "__main__":
    main()
