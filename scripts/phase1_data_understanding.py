"""
Phase 1: Data Understanding.

Loads the real M5 data, validates schema, and computes the summary
statistics that inform every later phase decision (target zero-rate for
Phase 6's model-family choice, series-volume spread for Phase 7 sample
weighting, price/calendar coverage for feature engineering in Phase 7).

Run: python scripts/phase1_data_understanding.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader, HIERARCHY_ID_COLS  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "interim" / "phase1_summary.md"


def main():
    loader = M5DataLoader(data_dir=str(DATA_DIR))
    data = loader.load(verbose=True)

    print("\n--- Schema validation ---")
    results = M5DataLoader.validate_schema(data)
    for name, r in results.items():
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {name}: {r['detail']}")
    n_failed = sum(1 for r in results.values() if not r["passed"])

    day_cols = data.day_cols
    sales_vals = data.sales[day_cols].to_numpy()

    # --- Hierarchy cardinalities ---
    hier = {c: data.sales[c].nunique() for c in HIERARCHY_ID_COLS}

    # --- Zero-sales / intermittency ---
    zero_pct_overall = float((sales_vals == 0).mean() * 100)
    zero_pct_per_series = (sales_vals == 0).mean(axis=1) * 100
    zero_pct_by_cat = (
        pd.DataFrame({"cat_id": data.sales["cat_id"],
                       "zero_pct": zero_pct_per_series})
        .groupby("cat_id")["zero_pct"].mean()
    )

    # --- Volume heterogeneity (mean daily units per series) ---
    mean_per_series = sales_vals.mean(axis=1)
    volume_summary = pd.Series(mean_per_series).describe()

    # --- Missingness ---
    sales_nulls = int(data.sales[day_cols].isnull().sum().sum())
    calendar_nulls = data.calendar.isnull().sum()
    price_rows = len(data.prices)
    price_nulls = int(data.prices["sell_price"].isnull().sum())

    # --- Price coverage: fraction of (item, store, day) with a known price ---
    # A missing price for a given wm_yr_wk means the item was not yet (or no
    # longer) sold at that store — legitimate, not corrupted data. We check
    # the rate here, not to "fix" it, but so Phase 7 feature engineering
    # knows how common this is and designs the price feature accordingly.
    n_item_store_pairs_with_price = data.prices.groupby(["item_id", "store_id"]).ngroups
    n_item_store_pairs_in_sales = data.sales.groupby(["item_id", "store_id"]).ngroups

    # --- Date range / horizon feasibility ---
    date_min, date_max = data.calendar["date"].min(), data.calendar["date"].max()
    n_calendar_days = len(data.calendar)
    n_sales_days = len(day_cols)
    spare_days_for_holdout = n_calendar_days - n_sales_days  # calendar extends past sales

    report_lines = []
    report_lines.append("# Phase 1 — Data Understanding: Summary Report\n")
    report_lines.append(f"Generated from real data in `{DATA_DIR}`.\n")

    report_lines.append("## Schema validation\n")
    report_lines.append(f"{len(results) - n_failed}/{len(results)} checks passed.\n")
    for name, r in results.items():
        report_lines.append(f"- **{name}**: {'PASS' if r['passed'] else 'FAIL'} — {r['detail']}")

    report_lines.append("\n## Hierarchy cardinalities (verified, not assumed)\n")
    for k, v in hier.items():
        report_lines.append(f"- `{k}`: {v} unique values")
    report_lines.append(f"- total series (item x store): {data.sales.shape[0]}")

    report_lines.append("\n## Time span\n")
    report_lines.append(f"- Calendar: {date_min.date()} to {date_max.date()} "
                         f"({n_calendar_days} days)")
    report_lines.append(f"- Sales history: {n_sales_days} days ({day_cols[0]}..{day_cols[-1]})")
    report_lines.append(f"- Calendar extends {spare_days_for_holdout} days beyond the sales "
                         f"history (post-competition future calendar, no sales — not usable "
                         f"as held-out ground truth, only as a features table if we ever needed "
                         f"future calendar features).")

    report_lines.append("\n## Intermittency (zero-sales days)\n")
    report_lines.append(f"- Overall zero-sales rate across all item-store-days: "
                         f"{zero_pct_overall:.1f}%")
    report_lines.append("- Mean zero-sales rate by category:")
    for cat, pct in zero_pct_by_cat.items():
        report_lines.append(f"  - {cat}: {pct:.1f}%")

    report_lines.append("\n## Volume heterogeneity (mean daily units per series)\n")
    report_lines.append("```")
    report_lines.append(str(volume_summary))
    report_lines.append("```")

    report_lines.append("\n## Missingness\n")
    report_lines.append(f"- Null values in sales day-columns: {sales_nulls}")
    report_lines.append(f"- Null `sell_price` rows: {price_nulls} / {price_rows} "
                         f"({price_nulls / price_rows * 100:.2f}%)")
    report_lines.append("- Calendar column null counts:")
    for col, n in calendar_nulls.items():
        if n > 0:
            report_lines.append(f"  - {col}: {n} ({n / n_calendar_days * 100:.1f}%)")

    report_lines.append("\n## Price table coverage\n")
    report_lines.append(f"- Item-store pairs with at least one price row: "
                         f"{n_item_store_pairs_with_price}")
    report_lines.append(f"- Item-store pairs present in the sales table: "
                         f"{n_item_store_pairs_in_sales}")

    report_text = "\n".join(report_lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report_text)
    print("\n" + report_text)
    print(f"\nReport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
