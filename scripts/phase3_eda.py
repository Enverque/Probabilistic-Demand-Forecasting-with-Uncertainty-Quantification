"""
Phase 3: Time-Series EDA.

Runs distributional analysis (reused DistributionAnalyzer) and new
trend/seasonality/holiday/price/structural-break analysis across multiple
hierarchy levels: Total, a few representative Stores, and a stratified
item-level sample (full item-level distribution fitting on all 30,490
series would be needlessly slow for an EDA phase; stratified sampling by
category, as the prototype did, is the right call here).

Run: python scripts/phase3_eda.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.data.hierarchy import build_hierarchy  # noqa: E402
from src.eda.distribution_analysis import DistributionAnalyzer  # noqa: E402
from src.eda.temporal_patterns import (  # noqa: E402
    linear_trend, weekly_seasonality, monthly_seasonality,
    holiday_event_effect, price_sales_correlation, structural_break_scan,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "interim" / "phase3_summary.md"
RANDOM_STATE = 42
ITEM_SAMPLE_PER_CAT = 700  # stratified sample size


def main():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    hierarchy = build_hierarchy(data.sales, data.day_cols)
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)  # align to sales history only
    dow = cal["date"].dt.dayofweek.to_numpy()
    month = cal["date"].dt.month.to_numpy()
    has_event = ((cal["event_name_1"].notna()) | (cal["event_name_2"].notna())).astype(int).to_numpy()

    lines = ["# Phase 3 — Time-Series EDA: Summary Report\n"]

    # ---- 1. Trend + seasonality at Total, State, and 2 representative Store levels ----
    lines.append("## Trend & seasonality by hierarchy level\n")
    levels_to_check = {
        "Total": hierarchy["Level1_total"].iloc[0][data.day_cols].to_numpy(dtype=float),
    }
    for _, row in hierarchy["Level3_store"].iterrows():
        levels_to_check[f"Store {row['store_id']}"] = row[data.day_cols].to_numpy(dtype=float)

    for name, vals in levels_to_check.items():
        trend = linear_trend(vals)
        weekly = weekly_seasonality(vals, dow)
        monthly = monthly_seasonality(vals, month)
        holiday = holiday_event_effect(vals, has_event)
        lines.append(f"### {name}")
        lines.append(f"- Trend: {trend['slope_per_day']:+.3f} units/day "
                      f"(R²={trend['r_squared']:.3f}, p={trend['p_value']:.2e})")
        lines.append(f"- Weekly seasonality: F={weekly['f_statistic']:.1f}, p={weekly['p_value']:.2e}, "
                      f"peak=dow{weekly['peak_dow']}, trough=dow{weekly['trough_dow']}, "
                      f"peak/trough={weekly['peak_to_trough_ratio']:.2f}x")
        lines.append(f"- Monthly seasonality: F={monthly['f_statistic']:.1f}, p={monthly['p_value']:.2e}")
        lines.append(f"- Holiday/event effect: event mean={holiday['event_mean']:.1f} vs. "
                      f"non-event mean={holiday['non_event_mean']:.1f} "
                      f"({holiday['pct_lift']:+.1f}%, p={holiday['p_value']:.2e})")

    # ---- 2. Structural breaks at Total level ----
    total_vals = levels_to_check["Total"]
    breaks = structural_break_scan(total_vals)
    lines.append("\n## Structural break scan (Total level)\n")
    lines.append(f"- Candidate breaks (|z|>2): {breaks['n_candidates']}")
    if breaks["top_break_day"] is not None:
        break_date = cal.loc[breaks["top_break_day"], "date"].date()
        lines.append(f"- Largest candidate: day index {breaks['top_break_day']} "
                      f"({break_date}), z={breaks['top_break_z']:+.2f}")

    # ---- 3. Distributional analysis: stratified item-level sample ----
    lines.append("\n## Distributional analysis (stratified item-level sample)\n")
    sample = (data.sales.groupby("cat_id", group_keys=False)[data.sales.columns]
              .apply(lambda g: g.sample(min(len(g), ITEM_SAMPLE_PER_CAT), random_state=RANDOM_STATE)))
    series_dict = {row["id"]: row[data.day_cols].to_numpy(dtype=float)
                   for _, row in sample.iterrows()}
    analyzer = DistributionAnalyzer()
    dist_df = analyzer.batch_analyze(series_dict, verbose=False)
    dist_df = dist_df.merge(sample[["id", "cat_id"]], left_on="series_id", right_on="id")

    lines.append(f"- Sample size: {len(dist_df)} series ({ITEM_SAMPLE_PER_CAT}/category, "
                 f"actual counts: {sample.groupby('cat_id').size().to_dict()})")
    lines.append(f"- Zero-inflated (ratio>1.5 vs. Poisson null): "
                 f"{dist_df['zi_is_zero_inflated'].mean()*100:.1f}% of series")
    lines.append(f"- Heavy-tailed: {dist_df['ht_is_heavy_tailed'].mean()*100:.1f}% of series")
    lines.append("- Intermittency classification (Syntetos-Boylan):")
    for cls, pct in (dist_df["int_classification"].value_counts(normalize=True) * 100).items():
        lines.append(f"  - {cls}: {pct:.1f}%")
    lines.append("- Best-fitting distribution (by AIC), overall:")
    for dist_name, pct in (dist_df["best_dist"].value_counts(normalize=True) * 100).items():
        lines.append(f"  - {dist_name}: {pct:.1f}%")
    lines.append("- Best-fitting distribution by category:")
    for cat, grp in dist_df.groupby("cat_id"):
        top = grp["best_dist"].value_counts(normalize=True) * 100
        lines.append(f"  - {cat}: " + ", ".join(f"{k}={v:.0f}%" for k, v in top.head(3).items()))

    # ---- 4. Price-sales relationship, same sample ----
    lines.append("\n## Price-sales relationship (stratified item-level sample)\n")
    # Filter prices down to only the sampled item-store pairs BEFORE
    # joining to the daily calendar. Joining the full 6.8M-row weekly
    # price table to every calendar day first (47M+ intermediate rows)
    # is what actually OOM-killed the first run of this script — the
    # fix is ordering the filter before the expensive operation, not
    # buying more memory.
    sample_key_df = sample[["item_id", "store_id"]].drop_duplicates()
    prices_filtered = data.prices.merge(sample_key_df, on=["item_id", "store_id"], how="inner")
    prices_wide = (prices_filtered.merge(data.calendar[["wm_yr_wk", "d"]].drop_duplicates(), on="wm_yr_wk")
                   .pivot_table(index=["item_id", "store_id"], columns="d", values="sell_price"))
    corr_results = []
    for _, row in sample.iterrows():
        key = (row["item_id"], row["store_id"])
        if key not in prices_wide.index:
            continue
        price_series = prices_wide.loc[key].reindex(data.day_cols).to_numpy(dtype=float)
        sales_series = row[data.day_cols].to_numpy(dtype=float)
        res = price_sales_correlation(price_series, sales_series)
        res["id"] = row["id"]
        corr_results.append(res)
    corr_df = pd.DataFrame(corr_results).dropna(subset=["spearman_r"])
    lines.append(f"- Series with usable price variation: {len(corr_df)}/{len(sample)}")
    lines.append(f"- Mean Spearman r (price vs. sales): {corr_df['spearman_r'].mean():+.3f}")
    lines.append(f"- % of series with negative correlation (expected direction): "
                 f"{(corr_df['spearman_r'] < 0).mean()*100:.1f}%")
    lines.append(f"- % with statistically significant correlation (p<0.05): "
                 f"{(corr_df['p_value'] < 0.05).mean()*100:.1f}%")

    report = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report)
    print(report)
    print(f"\nReport written to {OUT_PATH}")


if __name__ == "__main__":
    main()
