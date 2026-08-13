"""
Phase 10: Global vs. Local Models.

Uses the same 600-series sample/features as Phases 7-9, single window
(train through 2016-04-24, 28-day test -- consistent with Phase 8/9 for
comparability). Two comparisons:

1. Normal-history series: global (one model, all series pooled) vs. local
   (one LightGBM per series) on series that all have full training history.
2. Engineered cold-start: 50 randomly chosen series have their training
   data truncated to the last 60 days only (simulating a newly launched
   item), while the other 550 keep full history and the global model still
   trains on everyone pooled. This directly tests the claim Phase 0 asks
   about: does a global model's ability to borrow strength across series
   actually help on genuinely data-poor series, or is that just textbook
   folklore -- measured here, not assumed.

Run: python scripts/phase10_global_vs_local.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.features.engineering import build_long_format, add_features, feature_columns  # noqa: E402
from src.models.local_global import (  # noqa: E402
    fit_global_point_model, predict_global, fit_local_point_models, predict_local, MIN_LOCAL_TRAIN_ROWS,
)
from src.evaluation.metrics import wmape

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 200
HORIZON = 28
TRAIN_END = pd.Timestamp("2016-04-24")
TEST_END = pd.Timestamp("2016-05-22")
N_COLD_START_SERIES = 50
COLD_START_HISTORY_DAYS = 60


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))

    all_ids = sample["id"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    cold_start_ids = set(rng.choice(all_ids, size=N_COLD_START_SERIES, replace=False))
    print(f"{len(cold_start_ids)} series designated cold-start "
          f"(training history truncated to last {COLD_START_HISTORY_DAYS} days)")

    long_df = build_long_format(sample, data.day_cols, data.calendar, data.prices)

    # CRITICAL: truncate cold-start series' raw rows BEFORE feature
    # engineering, not after. A first version of this script truncated
    # post-hoc (after add_features), and inspection showed lag_28/
    # roll_mean_28 on the retained rows still fully reflected each item's
    # real 2011-2016 history -- add_features had already baked in years
    # of real prior demand before the truncation step ever ran, so the
    # "cold start" wasn't actually cold. Truncating the raw rows first
    # means groupby('id').shift()/rolling() genuinely can't see beyond
    # the retained window for these ids, which is what a real new-item
    # launch would look like.
    cold_cutoff = TRAIN_END - pd.Timedelta(days=COLD_START_HISTORY_DAYS)
    is_cold_id = long_df["id"].isin(cold_start_ids)
    long_df = long_df[~(is_cold_id & (long_df["date"] <= cold_cutoff))]

    featured = add_features(long_df, horizon=HORIZON)
    numeric_cols = featured.select_dtypes(include=["float64"]).columns
    featured[numeric_cols] = featured[numeric_cols].astype("float32")
    cols = feature_columns(HORIZON)

    train = featured[featured["date"] <= TRAIN_END].dropna(subset=cols)
    test = featured[(featured["date"] > TRAIN_END) & (featured["date"] <= TEST_END)].dropna(subset=cols)
    n_cold_test_rows = test["id"].isin(cold_start_ids).sum()
    print(f"Train rows (post cold-start truncation + feature dropna): {len(train)}")
    print(f"Test rows: {len(test)} ({n_cold_test_rows} from cold-start series -- "
          f"some cold-start test rows may be dropped here too if their features "
          f"are still NaN even with the full 60+28 day retained window)")

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        global_model = fit_global_point_model(train, cols, "sales", random_state=RANDOM_STATE)
    global_preds = predict_global(global_model, test, cols)
    print(f"Global model trained in {time.time()-t0:.0f}s")

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        local_models = fit_local_point_models(train, cols, "sales", random_state=RANDOM_STATE)
    local_preds = predict_local(local_models, test, cols)
    n_fallback_total = sum(1 for m in local_models.values() if isinstance(m, float))
    n_fallback_cold = sum(1 for sid, m in local_models.items() if isinstance(m, float) and sid in cold_start_ids)
    print(f"Local models trained in {time.time()-t0:.0f}s "
          f"({n_fallback_total}/{len(local_models)} series fell back to a constant overall, "
          f"{n_fallback_cold}/{N_COLD_START_SERIES} among the cold-start series -- "
          f"MIN_LOCAL_TRAIN_ROWS={MIN_LOCAL_TRAIN_ROWS})")

    is_cold_test = test["id"].isin(cold_start_ids).to_numpy()

    print("\n--- Normal-history series (550 series with full training data) ---")
    normal_mask = ~is_cold_test
    global_wmape_normal = wmape(test.loc[normal_mask, "sales"], global_preds.loc[normal_mask])
    local_wmape_normal = wmape(test.loc[normal_mask, "sales"], local_preds.loc[normal_mask])
    print(f"Global model WMAPE: {global_wmape_normal*100:.2f}%")
    print(f"Local model WMAPE:  {local_wmape_normal*100:.2f}%")

    print(f"\n--- Cold-start series ({N_COLD_START_SERIES} series, "
          f"{COLD_START_HISTORY_DAYS}-day truncated history) ---")
    cold_mask = is_cold_test
    global_wmape_cold = wmape(test.loc[cold_mask, "sales"], global_preds.loc[cold_mask])
    local_wmape_cold = wmape(test.loc[cold_mask, "sales"], local_preds.loc[cold_mask])
    print(f"Global model WMAPE: {global_wmape_cold*100:.2f}%")
    print(f"Local model WMAPE:  {local_wmape_cold*100:.2f}%")
    print(f"Local model's relative degradation on cold-start vs. normal series: "
          f"{(local_wmape_cold/local_wmape_normal - 1)*100:+.1f}%")
    print(f"Global model's relative degradation on cold-start vs. normal series: "
          f"{(global_wmape_cold/global_wmape_normal - 1)*100:+.1f}%")


if __name__ == "__main__":
    main()
