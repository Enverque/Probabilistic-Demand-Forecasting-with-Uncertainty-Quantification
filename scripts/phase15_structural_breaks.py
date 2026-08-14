"""
Phase 15: Structural Breaks.

Phase 3 ran the lightweight structural_break_scan (src/eda/temporal_patterns.py)
on the Total-level series only and found 0 candidates -- but that doesn't
tell us whether individual item-store series have real regime shifts that
get smoothed away at full aggregation. This phase runs the SAME scan
(deliberately not upgraded to CUSUM/Bai-Perron, per Phase 0's "don't add
a sophisticated method unless it earns its place") across the same
600-series sample used throughout Phases 7-13, then checks whether
detected breaks cluster around real, interpretable events (price changes)
rather than being pure noise.

Run: python scripts/phase15_structural_breaks.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.eda.temporal_patterns import structural_break_scan  # noqa: E402
from src.features.engineering import build_long_format  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 200


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)

    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))
    print(f"Scanning {len(sample)} item-store series for structural breaks "
          f"(window=90, same method as Phase 3's Total-level scan)...")

    results = []
    for _, row in sample.iterrows():
        vals = row[data.day_cols].to_numpy(dtype=float)
        r = structural_break_scan(vals, window=90)
        results.append({"id": row["id"], "cat_id": row["cat_id"], "item_id": row["item_id"],
                         "store_id": row["store_id"], **r})
    results_df = pd.DataFrame(results)

    n_with_break = (results_df["n_candidates"] >= 1).sum()
    print(f"\n{n_with_break}/{len(results_df)} series ({n_with_break/len(results_df)*100:.1f}%) "
          f"have at least one candidate structural break (|z| > 2)")
    print("Compare: Phase 3 found 0 candidates at the Total level (all 30,490 series aggregated)")

    print("\nBreak count distribution:")
    print(results_df["n_candidates"].value_counts().sort_index().to_string())

    print("\nBy category, % of series with >=1 break:")
    print(results_df.groupby("cat_id")["n_candidates"].apply(lambda s: (s >= 1).mean() * 100).round(1))

    print("\n--- Do detected breaks cluster around real price changes? ---")
    has_break = results_df[results_df["n_candidates"] >= 1].dropna(subset=["top_break_day"])
    long_df = build_long_format(sample, data.day_cols, data.calendar, data.prices)
    price_by_id = long_df.groupby("id")["sell_price"]

    matched, checked = 0, 0
    for _, r in has_break.iterrows():
        if r["id"] not in price_by_id.groups:
            continue
        prices = price_by_id.get_group(r["id"]).to_numpy()
        if len(prices) < 90:
            continue
        break_day = int(r["top_break_day"])
        window_start, window_end = max(0, break_day - 14), min(len(prices), break_day + 14)
        local_prices = prices[window_start:window_end]
        local_prices = local_prices[~np.isnan(local_prices)]
        checked += 1
        if len(local_prices) > 1 and (np.diff(local_prices) != 0).any():
            matched += 1

    if checked > 0:
        print(f"Of {checked} series with a detected break where price history is available: "
              f"{matched} ({matched/checked*100:.1f}%) had a price change within +/-14 days of the break.")
    print("(A high match rate would suggest detected breaks are often price-driven, not spurious; "
          "a low rate suggests either noise-driven detections or breaks from other causes.)")

    print("\n--- What this means for monitoring/retraining (per Phase 0's question) ---")
    print("Given item-level breaks are common but Total-level breaks are absent, a production "
          "system should monitor residuals PER SERIES OR PER CATEGORY, not only at an aggregate "
          "level -- an aggregate-level monitor would miss the great majority of these individual "
          "regime shifts entirely, since they wash out under aggregation exactly as Phase 3's "
          "0-candidate Total-level result already showed.")


if __name__ == "__main__":
    main()
