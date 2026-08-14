"""
Phase 14: Hierarchical Reconciliation.

Scoped to the FULL 30,490-series hierarchy using seasonal-naive (reusing
Phase 11's vectorized wide-array implementation) rather than the 600-
series ML sample used in Phases 7-13. This is a deliberate scope choice,
not a downgrade: reconciliation is fundamentally about whether summing
BOTTOM-LEVEL forecasts up to a real aggregate matches (or beats) a
forecast made DIRECTLY at that aggregate -- and our 600-series sample's
"bottom-up sum" would only cover ~2% of each store's real item count, so
it could never sum to that store's real total. Only the full-scale data
lets this comparison mean anything.

Two forecasts computed at every one of the 12 levels:
  - DIRECT: seasonal-naive applied straight to that level's own aggregated
    history (same as Phase 11's approach).
  - BOTTOM-UP: seasonal-naive applied ONLY at Level12 (item-store), then
    summed up according to each level's grouping -- coherent by
    construction, and exactly what Phase 2 showed a DIRECT approach is NOT.

Run: python scripts/phase14_reconciliation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy, HIERARCHY_LEVELS  # noqa: E402
from src.evaluation.wrmsse import rmsse, wrmsse_for_level, full_wrmsse, AGG_LEVEL_COLUMN_MAP  # noqa: E402
from src.evaluation.metrics import wmape  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase11_wrmsse import seasonal_naive_wide, weight_lookup_for_level  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HORIZON = 28
TRAIN_END = pd.Timestamp("2016-04-24")


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)
    train_end_idx = cal[cal["date"] == TRAIN_END].index[0]
    test_idx = list(range(train_end_idx + 1, train_end_idx + 1 + HORIZON))
    dow = cal["date"].dt.dayofweek.to_numpy()

    hierarchy = build_hierarchy(data.sales, data.day_cols)

    level12_wide = hierarchy["Level12_item_store"][data.day_cols].to_numpy(dtype=np.float32)
    level12_train = level12_wide[:, :train_end_idx + 1]
    level12_preds = seasonal_naive_wide(level12_train, train_end_idx, HORIZON, dow, agg="median")
    level12_keys = data.sales[["item_id", "store_id", "dept_id", "cat_id", "state_id"]].reset_index(drop=True)

    print("Level12 forecasts computed once. Building bottom-up sums for all 12 levels...\n")

    level_scores_direct = {}
    level_scores_bottomup = {}
    wmape_direct, wmape_bottomup = {}, {}

    for level_name in HIERARCHY_LEVELS:
        group_cols = AGG_LEVEL_COLUMN_MAP[level_name]
        agg = hierarchy[level_name]
        wide = agg[data.day_cols].to_numpy(dtype=np.float32)
        train_wide = wide[:, :train_end_idx + 1]
        actual_wide = wide[:, test_idx]

        direct_preds = seasonal_naive_wide(train_wide, train_end_idx, HORIZON, dow, agg="median")

        if len(group_cols) == 0:
            bu_preds = level12_preds.sum(axis=0, keepdims=True)
            keys = pd.Series(["Total"])
        elif len(group_cols) == 1:
            bu_df = pd.DataFrame(level12_preds, index=level12_keys[group_cols[0]])
            bu_grouped = bu_df.groupby(level=0).sum()
            keys = agg[group_cols[0]].reset_index(drop=True)
            bu_preds = bu_grouped.reindex(keys.values).to_numpy()
        else:
            multi_idx = pd.MultiIndex.from_arrays([level12_keys[group_cols[0]], level12_keys[group_cols[1]]])
            bu_df = pd.DataFrame(level12_preds, index=multi_idx)
            bu_grouped = bu_df.groupby(level=[0, 1]).sum()
            keys = pd.Series(list(zip(agg[group_cols[0]], agg[group_cols[1]])))
            bu_preds = bu_grouped.reindex(keys.values).to_numpy()

        w = weight_lookup_for_level(level_name, data.weights)
        weight_map = dict(zip(w["key"], w["weight"]))
        weights_series = keys.map(weight_map)
        assert weights_series.isna().sum() == 0, f"{level_name}: unmatched weights"

        rmsse_direct = pd.Series([rmsse(actual_wide[i], direct_preds[i], train_wide[i]) for i in range(wide.shape[0])])
        rmsse_bu = pd.Series([rmsse(actual_wide[i], bu_preds[i], train_wide[i]) for i in range(wide.shape[0])])
        level_scores_direct[level_name] = wrmsse_for_level(rmsse_direct, weights_series)
        level_scores_bottomup[level_name] = wrmsse_for_level(rmsse_bu, weights_series)

        wmape_direct[level_name] = wmape(actual_wide.flatten(), direct_preds.flatten())
        wmape_bottomup[level_name] = wmape(actual_wide.flatten(), bu_preds.flatten())

        print(f"{level_name:28s} DIRECT: WRMSSE={level_scores_direct[level_name]:.4f} "
              f"WMAPE={wmape_direct[level_name]*100:5.1f}%  |  "
              f"BOTTOM-UP: WRMSSE={level_scores_bottomup[level_name]:.4f} "
              f"WMAPE={wmape_bottomup[level_name]*100:5.1f}%")

    print(f"\nFull WRMSSE, DIRECT (independent forecast per level):    {full_wrmsse(level_scores_direct):.4f}")
    print(f"Full WRMSSE, BOTTOM-UP (reconciled, summed from Level12): {full_wrmsse(level_scores_bottomup):.4f}")

    print("\n--- Coherence check: do Store-level bottom-up forecasts sum EXACTLY to State-level? ---")
    store_bu_df = pd.DataFrame(level12_preds, index=level12_keys["store_id"])
    store_bu = store_bu_df.groupby(level=0).sum()

    store_state_map = data.sales[["store_id", "state_id"]].drop_duplicates().set_index("store_id")["state_id"]
    state_keys_for_l12 = level12_keys["store_id"].map(store_state_map)
    state_bu_df = pd.DataFrame(level12_preds, index=state_keys_for_l12)
    state_bu = state_bu_df.groupby(level=0).sum()

    store_bu_with_state = store_bu.copy()
    store_bu_with_state["state_id"] = store_bu_with_state.index.map(store_state_map)
    reconciled_state_from_store = store_bu_with_state.groupby("state_id").sum(numeric_only=True)
    max_gap = (reconciled_state_from_store - state_bu).abs().to_numpy().max()
    print(f"Max absolute gap, State bottom-up vs. (sum of Store bottom-up): {max_gap:.10f} "
          f"(should be ~0 -- coherent BY CONSTRUCTION)")

    print("\nFor comparison, recall Phase 2's finding using DIRECT (non-reconciled, median-based) "
          "forecasts: max gap was 0.81% of the state total -- a real, non-trivial incoherence "
          "that bottom-up reconciliation eliminates completely, as shown above.")


if __name__ == "__main__":
    main()
