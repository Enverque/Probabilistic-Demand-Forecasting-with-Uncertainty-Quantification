"""
Phase 18: Forecast Dashboard.

A Streamlit app for a Store Manager / Inventory Planner: pick a slice of
the hierarchy (state/store/category/department/item, any combination),
see historical demand, a 28-day quantile forecast with uncertainty bands,
calibration context, and a newsvendor reorder recommendation with
estimated cost.

IMPORTANT SCOPE NOTE (shown in-app too, not just here): this dashboard
uses the FAST quantile seasonal-naive forecaster
(src/forecasting/dashboard_forecaster.py), not Phase 7's LightGBM model.
Retraining LightGBM per user click isn't fast enough for an interactive
app; the accuracy/calibration numbers measured in Phases 7-13 do NOT
transfer to this dashboard's forecasts, and the app says so rather than
implying the same validated performance.

Run: streamlit run app/dashboard.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.loader import M5DataLoader  # noqa: E402
from src.forecasting.dashboard_forecaster import quantile_seasonal_naive  # noqa: E402
from src.inventory.newsvendor import critical_ratio, order_quantity_from_quantiles, simulate_period  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
HORIZON = 28
DASHBOARD_QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]

PHASE13_CALIBRATION = pd.DataFrame({
    "interval": ["50%", "80%", "95%"],
    "nominal": [0.50, 0.80, 0.95],
    "empirical (Phase 13, ML model)": [0.743, 0.917, 0.975],
})


@st.cache_data
def load_data():
    data = M5DataLoader(data_dir=str(DATA_DIR)).load(verbose=False)
    return data


def main():
    st.set_page_config(page_title="M5 Demand Forecast Dashboard", layout="wide")
    st.title("Demand Forecast & Reorder Dashboard")
    st.caption(
        "Scope note: forecasts here use a fast quantile seasonal-naive method "
        "(src/forecasting/dashboard_forecaster.py), NOT the LightGBM quantile model "
        "validated in Phases 7-13. Retraining that model per selection isn't fast enough "
        "for an interactive app. Phase 13's calibration numbers (shown below for reference) "
        "were measured on the LightGBM model and do not describe this dashboard's own forecasts."
    )

    data = load_data()
    cal = data.calendar.iloc[:len(data.day_cols)].reset_index(drop=True)
    dow_hist = cal["date"].dt.dayofweek.to_numpy()
    future_dates = pd.date_range(cal["date"].iloc[-1] + pd.Timedelta(days=1), periods=HORIZON)
    dow_full = np.concatenate([dow_hist, future_dates.dayofweek.to_numpy()])

    st.sidebar.header("Select a hierarchy slice")
    sales = data.sales
    state = st.sidebar.selectbox("State", ["All"] + sorted(sales["state_id"].unique()))
    filtered = sales if state == "All" else sales[sales["state_id"] == state]
    store = st.sidebar.selectbox("Store", ["All"] + sorted(filtered["store_id"].unique()))
    filtered = filtered if store == "All" else filtered[filtered["store_id"] == store]
    cat = st.sidebar.selectbox("Category", ["All"] + sorted(filtered["cat_id"].unique()))
    filtered = filtered if cat == "All" else filtered[filtered["cat_id"] == cat]
    dept = st.sidebar.selectbox("Department", ["All"] + sorted(filtered["dept_id"].unique()))
    filtered = filtered if dept == "All" else filtered[filtered["dept_id"] == dept]
    item = st.sidebar.selectbox("Item", ["All"] + sorted(filtered["item_id"].unique()))
    filtered = filtered if item == "All" else filtered[filtered["item_id"] == item]

    if filtered.empty:
        st.error("No series match this selection.")
        return

    series = filtered[data.day_cols].sum(axis=0).to_numpy(dtype=float)
    n_series_aggregated = len(filtered)
    st.write(f"**Selection aggregates {n_series_aggregated} underlying item-store series.**")

    forecast = quantile_seasonal_naive(series, dow_full, HORIZON, DASHBOARD_QUANTILES, n_periods=8)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Historical demand + 28-day forecast")
        hist_tail = 120
        hist_df = pd.DataFrame({"date": cal["date"].iloc[-hist_tail:], "value": series[-hist_tail:]})
        fc_df = pd.DataFrame({"date": future_dates, "value": forecast[0.5]})
        chart_df = pd.concat([hist_df, fc_df])
        st.line_chart(chart_df.set_index("date"))

        band_df = pd.DataFrame({
            "date": future_dates, "P10": forecast[0.1], "P25": forecast[0.25],
            "P50": forecast[0.5], "P75": forecast[0.75], "P90": forecast[0.9],
        })
        st.dataframe(band_df.set_index("date").round(2), height=250)

    with col2:
        st.subheader("Calibration reference (Phase 13)")
        st.dataframe(PHASE13_CALIBRATION.set_index("interval"))
        st.caption("Measured on the LightGBM model, item-store level. Included for context only.")

    st.subheader("Reorder recommendation (newsvendor)")
    c1, c2 = st.columns(2)
    stockout_cost = c1.slider("Stockout cost per unit ($)", 0.5, 20.0, 4.0, 0.5)
    holding_cost = c2.slider("Holding cost per unit ($)", 0.5, 20.0, 1.0, 0.5)

    cr = critical_ratio(stockout_cost, holding_cost)
    quantile_dict_28day = {q: float(np.sum(forecast[q])) for q in DASHBOARD_QUANTILES}
    order_qty = order_quantity_from_quantiles(quantile_dict_28day, cr)

    st.write(f"Critical ratio: **{cr:.2f}**  |  "
             f"Recommended 28-day order quantity: **{order_qty:.0f} units**")

    expected_demand = quantile_dict_28day[0.5]
    st.write(f"For reference: P50 (median) 28-day demand estimate is **{expected_demand:.0f} units**.")

    if st.checkbox("Show expected cost under a few demand scenarios"):
        scenario_rows = []
        for q in DASHBOARD_QUANTILES:
            scenario_demand = quantile_dict_28day[q]
            result = simulate_period(scenario_demand, order_qty, stockout_cost, holding_cost)
            scenario_rows.append({"scenario (demand quantile)": f"P{int(q*100)}",
                                   "assumed demand": round(scenario_demand, 1),
                                   "cost if this occurs": round(result["total_cost"], 2)})
        st.dataframe(pd.DataFrame(scenario_rows), hide_index=True)


if __name__ == "__main__":
    main()
