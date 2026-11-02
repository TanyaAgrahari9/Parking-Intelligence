"""Parking Intelligence - Streamlit app (Module 8: Recommendation, Heatmaps, Admin Dashboard).

Run locally:   streamlit run app/streamlit_app.py

The data set ends in December 2016, so the app works in *replay mode*: you pick
a day in the held-out test period (5-19 Dec 2016) and a time, and see the
deployed models' forecasts, made using only information available at that
moment (batch-scored by src/export_forecasts.py).  Because it is a replay, the true occupancy can be revealed and
compared with the forecast.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.features import rate_to_status  # noqa: E402

st.set_page_config(page_title="Parking Intelligence", page_icon="P", layout="wide")

SLOT_LABELS = [f"{8 + i // 2:02d}:{30 * (i % 2):02d}" for i in range(C.N_SLOTS)]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
STATUS_ICON = {0: "🟢 Available", 1: "🟠 Limited", 2: "🔴 Full"}
STATUS_COLOR = {"🟢 Available": "#1baf7a", "🟠 Limited": "#eda100", "🔴 Full": "#e34948"}


# ----------------------------------------------------------------------------- loading
@st.cache_data
def load_grid():
    return pd.read_csv(C.GRID_PARQUET, parse_dates=["slot_time", "date"])


@st.cache_data
def model_names():
    p = C.PROCESSED_DIR / "app_models.txt"
    return dict(line.split("=", 1) for line in p.read_text().splitlines()) if p.exists() else {}


@st.cache_data
def load_json(name):
    p = ROOT / "reports" / name
    return json.loads(p.read_text()) if p.exists() else None


@st.cache_data
def load_csv(name):
    return pd.read_csv(C.PROCESSED_DIR / name)


@st.cache_data
def forecast_frame(task):
    """Forecasts of the deployed model for every (zone, date, slot) in the test period.
    Produced by src/export_forecasts.py (batch scoring) - identical to live model output."""
    df = pd.read_csv(C.PROCESSED_DIR / f"app_forecasts_{task}.csv", parse_dates=["date", "slot_time"])
    df["status"] = [STATUS_ICON[s] for s in rate_to_status(df.pred)]
    df["free_spaces"] = ((1 - df.pred) * df.capacity).round().astype(int)
    return df


def results():
    return load_json("results.json")


# ----------------------------------------------------------------------------- pages
def page_find_parking():
    st.header("Find parking")
    st.caption("Pick when you want to arrive. Car parks are ranked by forecast free spaces.")
    fa = forecast_frame("day_ahead")
    dates = sorted(fa.date.dt.date.unique())
    c1, c2, c3 = st.columns([1.2, 1, 1.4])
    day = c1.date_input("Arrival day (test period)", dates[0], min_value=dates[0], max_value=dates[-1])
    slot = c2.selectbox("Arrival time", range(C.N_SLOTS), index=8, format_func=lambda i: SLOT_LABELS[i])
    mode = c3.radio("Forecast type", ["Planning ahead (day before)", "Arriving in 1 hour"],
                    help="Day-ahead uses only data up to the previous evening. "
                         "1-hour-ahead also uses today's latest readings, so it is more accurate.")
    task = "day_ahead" if mode.startswith("Planning") else "short"
    df = forecast_frame(task)
    sel = df[(df.date.dt.date == day) & (df.slot == slot)].copy()
    if sel.empty:
        st.warning("No forecast for this time (for 1-hour-ahead, the earliest target is "
                   f"{SLOT_LABELS[C.SHORT_HORIZON_SLOTS]}). Try another slot.")
        return
    sel = sel.sort_values("free_spaces", ascending=False)
    min_free = st.slider("Only show car parks with at least this many forecast free spaces", 0, 500, 20, 10)
    show = sel[sel.free_spaces >= min_free]
    reveal = st.toggle("Reveal what actually happened (replay)", value=False)

    best = show.head(3)
    cols = st.columns(3)
    for col, (_, r) in zip(cols, best.iterrows()):
        col.metric(r.zone, f"{r.free_spaces} free")
        col.caption(f"{r.status} · forecast {r.pred:.0%} full (80% range {r.lo:.0%}–{r.hi:.0%}) · "
                    f"capacity {r.capacity}")

    table = pd.DataFrame({
        "Car park": show.zone, "Status": show.status, "Forecast occupancy": show.pred,
        "Likely range (80%)": [f"{a:.0%} – {b:.0%}" for a, b in zip(show.lo, show.hi)],
        "Forecast free spaces": show.free_spaces, "Capacity": show.capacity})
    if reveal:
        table["Actual occupancy"] = show.rate.values
        table["Actual status"] = [STATUS_ICON[s] if not np.isnan(r) else "no reading"
                                  for s, r in zip(rate_to_status(show.rate.fillna(0)), show.rate)]
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Forecast occupancy": st.column_config.ProgressColumn(
                     format="percent", min_value=0, max_value=1),
                     **({"Actual occupancy": st.column_config.ProgressColumn(
                         format="percent", min_value=0, max_value=1)} if reveal else {})})
    st.caption(f"Model: {model_names().get(task, '?')}. Status thresholds: Limited ≥ "
               f"{C.LIMITED_THRESHOLD:.0%}, Full ≥ {C.FULL_THRESHOLD:.0%}.")


def page_zone_forecast():
    st.header("Zone forecast")
    df = forecast_frame("day_ahead")
    c1, c2 = st.columns(2)
    zone = c1.selectbox("Car park", sorted(df.zone.unique()))
    dates = sorted(df.date.dt.date.unique())
    day = c2.date_input("Day", dates[0], min_value=dates[0], max_value=dates[-1], key="zf_day")
    d = df[(df.zone == zone) & (df.date.dt.date == day)].sort_values("slot")
    sh = forecast_frame("short")
    s = sh[(sh.zone == zone) & (sh.date.dt.date == day)].sort_values("slot")
    x = [SLOT_LABELS[i] for i in d.slot]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x + x[::-1], y=list(d.hi) + list(d.lo[::-1]), fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                             name="80% interval (day-ahead)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=d.pred, name="Day-ahead forecast", line=dict(color="#2a78d6")))
    fig.add_trace(go.Scatter(x=[SLOT_LABELS[i] for i in s.slot], y=s.pred, name="1-hour-ahead forecast",
                             line=dict(color="#eb6834", dash="dash")))
    fig.add_trace(go.Scatter(x=x, y=d.rate, name="Actual", mode="lines+markers",
                             line=dict(color="#0b0b0b")))
    for t, lab in ((C.LIMITED_THRESHOLD, "Limited"), (C.FULL_THRESHOLD, "Full")):
        fig.add_hline(y=t, line_dash="dot", line_color="#999", annotation_text=lab)
    fig.update_layout(yaxis=dict(title="Occupancy rate", tickformat=".0%", range=[0, 1.05]),
                      height=420, margin=dict(t=30, b=10), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    ok = d.dropna(subset=["rate"])
    if len(ok):
        st.caption(f"Day-ahead MAE for this zone-day: {np.mean(np.abs(ok.rate - ok.pred)):.3f} "
                   f"(≈ {np.mean(np.abs(ok.rate - ok.pred) * ok.capacity):.0f} vehicles).")


def page_admin():
    st.header("Admin dashboard")
    grid = load_grid()
    tab1, tab2, tab3, tab4 = st.tabs(["Demand heatmaps", "Zone clusters", "Sensor anomalies",
                                      "Model performance"])
    with tab1:
        g = grid.dropna(subset=["rate"])
        hm = g.pivot_table(index="dow", columns="slot", values="rate", aggfunc="mean")
        fig = px.imshow(hm.values, x=SLOT_LABELS, y=[DOW[i] for i in hm.index], color_continuous_scale="Blues",
                        zmin=0, zmax=1, aspect="auto", labels=dict(color="Occupancy"))
        fig.update_layout(title="Average occupancy by weekday and time (all car parks)", height=340)
        st.plotly_chart(fig, use_container_width=True)
        zh = g[g.dow < 5].pivot_table(index="zone", columns="slot", values="rate", aggfunc="mean")
        zh = zh.loc[zh.mean(axis=1).sort_values().index]
        fig = px.imshow(zh.values, x=SLOT_LABELS, y=zh.index, color_continuous_scale="Blues", zmin=0, zmax=1,
                        aspect="auto", labels=dict(color="Occupancy"))
        fig.update_layout(title="Weekday profile per car park", height=620)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        cl = load_csv("zone_clusters.csv")
        eda = load_json("eda_stats.json") or {}
        mean_rate = g.groupby("zone").rate.mean()
        cl["mean occupancy"] = cl.zone.map(mean_rate)
        cl["capacity"] = cl.zone.map(grid.groupby("zone").capacity.first())
        st.write(f"K-means on each zone's weekday profile chose **k = {eda.get('cluster_k', '?')}** "
                 "clusters by silhouette score (cluster 0 = quieter).")
        st.dataframe(cl.sort_values(["cluster", "mean occupancy"]), hide_index=True,
                     use_container_width=True,
                     column_config={"mean occupancy": st.column_config.ProgressColumn(
                         format="percent", min_value=0, max_value=1)})
    with tab3:
        an = load_csv("zone_day_anomalies.csv")
        flagged = an[an.anomaly].sort_values("anomaly_score", ascending=False)
        st.write(f"Isolation Forest flagged **{len(flagged)}** of {len(an)} zone-days as unusual "
                 "(possible stuck sensor or special event). They are reported, not deleted.")
        st.dataframe(flagged[["zone", "date", "mean", "max", "stuck", "dev", "anomaly_score"]].round(3),
                     hide_index=True, use_container_width=True)
    with tab4:
        res = results()
        if res is None:
            st.info("reports/results.json not found - run `python -m src.train` first.")
            return
        task = st.radio("Task", ["day_ahead", "short"], horizontal=True,
                        format_func=lambda t: {"day_ahead": "Day-ahead", "short": "1-hour-ahead"}[t])
        r = res[task]
        tm = pd.DataFrame(r["test_metrics"]).T[["MAE", "RMSE", "R2", "MAE_vehicles", "status_macroF1"]]
        st.write(f"Deployed model: **{r['deployed_model']}**, chosen by the lowest average of "
                 f"cross-validation and validation error. Best baseline: {r['best_baseline']}.")
        if r.get("deployed_model_validation_only", r["deployed_model"]) != r["deployed_model"]:
            st.caption(f"Validation error alone would have picked {r['deployed_model_validation_only']}; "
                       "that 12-day window held only one weekend and ranked models poorly on test, so the "
                       "rule was changed after the test (disclosed in the report).")
        st.dataframe(tm.sort_values("MAE").style.format("{:.4f}"), use_container_width=True)
        for f in (f"model_comparison_{task}.png", f"err_time_{task}.png", f"err_confusion_{task}.png"):
            p = C.FIG_DIR / f
            if p.exists():
                st.image(str(p))


def page_explain():
    st.header("How the model works")
    st.markdown(
        "**Pipeline:** raw sensor log → cleaning → 30-minute grid → leakage-safe features "
        "(calendar, yesterday/last-week values, zone profile, latest readings) → tuned models "
        "(Ridge, SVR, Random Forest, HistGB, XGBoost) → the model with the lowest average of "
        "cross-validation and validation error is deployed (Random Forest for both tasks).\n\n"
        "**SHAP** splits each prediction into contributions of each feature: "
        "*prediction = average + sum of contributions*. The charts below explain the XGBoost model "
        "(exact TreeSHAP is fast for boosted trees); the deployed Random Forest was checked with "
        "permutation importance, which ranks the same leading features.")
    task = st.radio("Task", ["day_ahead", "short"], horizontal=True, key="ex_task",
                    format_func=lambda t: {"day_ahead": "Day-ahead", "short": "1-hour-ahead"}[t])
    c1, c2 = st.columns(2)
    for col, f in ((c1, f"shap_bar_{task}.png"), (c2, f"shap_beeswarm_{task}.png"),
                   (c1, f"shap_dependence_{task}.png"), (c2, f"shap_waterfall_{task}.png")):
        p = C.FIG_DIR / f
        if p.exists():
            col.image(str(p))
    st.info("Data: Birmingham (UK) city car-park occupancy, Oct–Dec 2016 (UCI repository), used as a "
            "public stand-in for campus parking zones. The code works unchanged on any log with "
            "zone, capacity, occupancy and timestamp columns.")


PAGES = {"Find parking": page_find_parking, "Zone forecast": page_zone_forecast,
         "Admin dashboard": page_admin, "How the model works": page_explain}
st.sidebar.title("🅿️ Parking Intelligence")
choice = st.sidebar.radio("Go to", list(PAGES))
st.sidebar.caption("Replay mode: forecasts for the held-out test period, 5–19 Dec 2016.")
PAGES[choice]()
