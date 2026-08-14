"""
Phase 11: WRMSSE.

Two parts, scoped honestly given what we actually have:

1. FULL, OFFICIAL WRMSSE for the seasonal-naive baseline, computed across
   all 12 hierarchy levels and all 30,490 base series -- feasible without
   full-scale ML training because seasonal-naive is pure historical
   averaging, vectorized directly on the wide (series x day) arrays rather
   than melting to long format (which OOM-killed earlier phases at far
   smaller scale). At each of the 12 levels, seasonal-naive is applied
   DIRECTLY to that level's own aggregated history -- matching how M5
   actually expects submissions (one forecast per level, not required to
   be bottom-up coherent), not a bottom-up sum of item-level forecasts.

2. A SAMPLE-SCOPE WRMSSE for Phase 7's quantile ML model, restricted to
   Level12 (item-store) on our 600-series sample only, with weights
   renormalized within the sample -- explicitly labeled as a partial
   approximation, not the official metric, since we never trained ML at
   full 30,490-series scale. Compared against the Level12-only slice of
   seasonal-naive's FULL run for as fair a comparison as this scope allows.

Run: python scripts/phase11_wrmsse.py
"""
import sys
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy, HIERARCHY_LEVELS  # noqa: E402
from src.evaluation.wrmsse import rmsse, wrmsse_for_level, full_wrmsse, AGG_LEVEL_COLUMN_MAP  # noqa: E402
from src.features.engineering import build_long_format, add_features, feature_columns  # noqa: E402
from src.forecasting.probabilistic import generate_probabilistic_forecast  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HORIZON = 28
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 200
TRAIN_END = pd.Timestamp("2016-04-24")
TEST_END = pd.Timestamp("2016-05-22")


def seasonal_naive_wide(wide: np.ndarray, train_end_idx: int, horizon: int,
                         dow: np.ndarray, n_periods: int = 4, agg: str = "mean") -> np.ndarray:
    """
    Same weekday-matching logic as Phase 11's version, but defaults to
    MEDIAN rather than mean. This matters specifically for this phase:
    mean is a linear operator, so a mean-based forecast is coherent by
    construction (sum-of-means == mean-of-sums), which is exactly what
    Phase 2 found and is what made the first run of THIS script show
    IDENTICAL direct and bottom-up scores at every single level -- a
    real result, but a degenerate one that can't answer Phase 0's actual
    question (does reconciliation help or hurt ACCURACY, not just
    coherence). Median doesn't distribute over sums, so switching to it
    here -- for both the Level12 base forecast and the "direct" per-level
    forecast -- lets bottom-up and direct genuinely diverge, the same fix
    Phase 2 made for the same underlying reason.
    """
    n_series = wide.shape[0]
    train_dow = dow[:train_end_idx + 1]
    reduce_fn = np.median if agg == "median" else np.mean
    weekday_vals: Dict[int, np.ndarray] = {}
    for wd in range(7):
        col_idxs = np.flatnonzero(train_dow == wd)
        last_n = col_idxs[-n_periods:] if len(col_idxs) > 0 else col_idxs
        if len(last_n) == 0:
            weekday_vals[wd] = reduce_fn(wide[:, :train_end_idx + 1], axis=1)
        else:
            weekday_vals[wd] = reduce_fn(wide[:, last_n], axis=1)

    preds = np.zeros((n_series, horizon))
    for h in range(horizon):
        target_dow = int(dow[train_end_idx + 1 + h])
        preds[:, h] = weekday_vals[target_dow]
    return preds


def weight_lookup_for_level(level_name: str, weights: pd.DataFrame) -> pd.DataFrame:
    level_id = level_name.split("_")[0]
    cols = AGG_LEVEL_COLUMN_MAP[level_name]
    w = weights[weights["Level_id"] == level_id][["Agg_Level_1", "Agg_Level_2", "weight"]].copy()
    if len(cols) == 0:
        w["key"] = "Total"
    elif len(cols) == 1:
        w["key"] = w["Agg_Level_1"]
    else:
        w["key"] = list(zip(w["Agg_Level_1"], w["Agg_Level_2"]))
    return w[["key", "weight"]]


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)
    train_end_idx = cal[cal["date"] == TRAIN_END].index[0]
    test_idx = list(range(train_end_idx + 1, train_end_idx + 1 + HORIZON))
    assert len(test_idx) == HORIZON
    dow = cal["date"].dt.dayofweek.to_numpy()

    hierarchy = build_hierarchy(data.sales, data.day_cols)

    print("=== PART 1: Full official WRMSSE, seasonal-naive, all 12 levels, all 30,490 series ===\n")
    level_scores = {}
    for level_name in HIERARCHY_LEVELS:
        # Use AGG_LEVEL_COLUMN_MAP's column order here, NOT HIERARCHY_LEVELS'
        # (src/data/hierarchy.py) -- they disagree for Level11 specifically
        # (item-then-state vs. the weights file's state-then-item), and
        # using the wrong one silently produced zero matches for all 9,147
        # Level11 series on the first run of this script (caught because
        # unmatched_weight=9147 printed for that level, which then fed a
        # false "perfect" 0.0 score into the overall average -- corrupting
        # it silently if unmatched counts hadn't been logged explicitly).
        group_cols = AGG_LEVEL_COLUMN_MAP[level_name]
        agg = hierarchy[level_name]
        wide = agg[data.day_cols].to_numpy(dtype=np.float32)
        train_wide = wide[:, :train_end_idx + 1]
        actual_wide = wide[:, test_idx]

        preds_wide = seasonal_naive_wide(train_wide, train_end_idx, HORIZON, dow)

        rmsse_vals = np.array([rmsse(actual_wide[i], preds_wide[i], train_wide[i])
                                for i in range(wide.shape[0])], dtype=object)
        rmsse_series = pd.Series([v if v is not None else np.nan for v in rmsse_vals], dtype=float)

        if len(group_cols) == 0:
            keys = pd.Series(["Total"])
        elif len(group_cols) == 1:
            keys = agg[group_cols[0]].reset_index(drop=True)
        else:
            keys = pd.Series(list(zip(agg[group_cols[0]], agg[group_cols[1]])))

        w = weight_lookup_for_level(level_name, data.weights)
        weight_map = dict(zip(w["key"], w["weight"]))
        weights_series = keys.map(weight_map)
        n_unmatched = weights_series.isna().sum()
        if n_unmatched > 0:
            raise RuntimeError(
                f"{level_name}: {n_unmatched}/{len(keys)} series failed to match a weight -- "
                f"treating these as weight=0 would silently understate this level's score. "
                f"Check AGG_LEVEL_COLUMN_MAP's column order against the weights file for this level.")

        level_score = wrmsse_for_level(rmsse_series, weights_series)
        level_scores[level_name] = level_score
        n_undefined_rmsse = rmsse_series.isna().sum()
        print(f"{level_name:28s} n_series={len(agg):6d}  weighted_RMSSE={level_score:.4f}  "
              f"undefined_scale={n_undefined_rmsse}  unmatched_weight={n_unmatched}")

    overall_wrmsse = full_wrmsse(level_scores)
    print(f"\nFULL OFFICIAL WRMSSE (seasonal-naive, all levels, all series): {overall_wrmsse:.4f}")

    print("\n=== PART 2: Sample-scope WRMSSE, quantile ML (Phase 7 model), Level12 only, 600-series sample ===\n")
    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))
    long_df = build_long_format(sample, data.day_cols, data.calendar, data.prices)
    featured = add_features(long_df, horizon=HORIZON)
    numeric_cols = featured.select_dtypes(include=["float64"]).columns
    featured[numeric_cols] = featured[numeric_cols].astype("float32")
    cols = feature_columns(HORIZON)
    train_ml = featured[featured["date"] <= TRAIN_END].dropna(subset=cols)
    test_ml = featured[(featured["date"] > TRAIN_END) & (featured["date"] <= TEST_END)].dropna(subset=cols)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds_ml = generate_probabilistic_forecast(train_ml, test_ml, cols, "sales",
                                                     quantiles=[0.5], random_state=RANDOM_STATE)
    test_ml = test_ml.reset_index(drop=True)
    preds_ml = preds_ml.reset_index(drop=True)

    sample_ids = sample["id"].tolist()
    w12 = weight_lookup_for_level("Level12_item_store", data.weights)
    w12_map = dict(zip(w12["key"], w12["weight"]))

    rmsse_rows = []
    for series_id in sample_ids:
        train_row = sample.loc[sample["id"] == series_id, data.day_cols].to_numpy(dtype=np.float32)[0]
        train_arr = train_row[:train_end_idx + 1]
        mask = (test_ml["id"] == series_id).to_numpy()
        if mask.sum() != HORIZON:
            continue
        actual_arr = test_ml.loc[mask, "sales"].to_numpy()
        pred_arr = preds_ml.loc[mask, 0.5].to_numpy()
        item_id = sample.loc[sample["id"] == series_id, "item_id"].iloc[0]
        store_id = sample.loc[sample["id"] == series_id, "store_id"].iloc[0]
        weight = w12_map.get((item_id, store_id), 0.0)
        r = rmsse(actual_arr, pred_arr, train_arr)
        rmsse_rows.append({"id": series_id, "rmsse": r if r is not None else np.nan, "weight": weight})

    rmsse_df = pd.DataFrame(rmsse_rows)
    total_weight = rmsse_df["weight"].sum()
    rmsse_df["weight_renorm"] = rmsse_df["weight"] / total_weight
    sample_wrmsse_ml = wrmsse_for_level(rmsse_df["rmsse"], rmsse_df["weight_renorm"])
    print(f"Sample-scope WRMSSE (quantile ML, P50, Level12, {len(rmsse_df)} series, "
          f"renormalized weights): {sample_wrmsse_ml:.4f}")
    print(f"[{rmsse_df['rmsse'].isna().sum()}/{len(rmsse_df)} series had undefined scale, excluded]")

    print(f"\nFor comparison, seasonal-naive's FULL Level12 contribution "
          f"(all 30,490 series, official weights): {level_scores['Level12_item_store']:.4f}")
    print("NOTE: these two numbers are NOT strictly apples-to-apples -- the ML number uses "
          "renormalized weights over 600 series, the seasonal-naive number uses official "
          "weights over all 30,490. Directionally comparable, not a precise head-to-head.")


if __name__ == "__main__":
    main()
