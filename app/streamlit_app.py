"""Parking Intelligence - Streamlit app (Module 8: Recommendation, Heatmaps, Admin Dashboard).

Run locally:   streamlit run app/streamlit_app.py

The data set ends in December 2016, so the app works in *replay mode*: you pick
a day in the held-out test period (5-19 Dec 2016) and a time, and see the
deployed models' forecasts, made using only information available at that
moment (batch-scored by src/export_forecasts.py). Because it is a replay, the
true occupancy can be revealed and compared with the forecast.
"""
import hashlib
import hmac
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config as C  # noqa: E402
from src.features import rate_to_status  # noqa: E402

st.set_page_config(page_title="Parking Intelligence", page_icon="🅿️", layout="wide",
                   initial_sidebar_state="collapsed")

# ----------------------------------------------------------------------------- colours and words
BG, CARD, LINE = "#0B1416", "#122126", "rgba(255,255,255,0.08)"
TEXT, MUTED = "#E8F2F0", "#8FA6A3"
TEAL, CORAL, VIOLET, SKY = "#2DD4BF", "#FB7D5B", "#A78BFA", "#60A5FA"
STATUS = {0: ("Available", "#34D399", "Plenty of space"),
          1: ("Limited", "#FBBF24", "Filling up"),
          2: ("Full", "#F87171", "Hard to find a space")}
HEAT = [[0, "#0E2A2C"], [0.35, "#0F766E"], [0.6, "#2DD4BF"], [0.8, "#FBBF24"], [1, "#F87171"]]
SLOT_LABELS = [f"{8 + i // 2:02d}:{30 * (i % 2):02d}" for i in range(C.N_SLOTS)]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
CLUE_WORDS = {"rate_t": "How full it is right now", "zone_profile": "How full it usually is at this time",
              "delta_t": "Filling up or emptying", "slot": "Time of day", "rate_t_1": "How full it was 30 min ago",
              "rate_t_2": "How full it was 1 hour ago", "lag_7d": "Same time last week",
              "lag_1d": "Same time yesterday", "days_to_christmas": "Days left until Christmas",
              "dow": "Day of the week", "roll7_same_slot": "Average of the last 7 days",
              "prevday_mean": "Yesterday's average", "today_mean_so_far": "Today's average so far",
              "zone": "Which car park", "hour": "Hour of the day", "capacity": "Size of the car park",
              "is_weekend": "Weekend or weekday"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html, body, .stApp, .stApp button, .stApp input, .stApp label, [data-testid="stMarkdownContainer"] {{
  font-family: 'Inter', system-ui, sans-serif; }}
.stApp {{ background:
  radial-gradient(900px 500px at 8% -10%, rgba(45,212,191,.13), transparent 60%),
  radial-gradient(800px 500px at 100% 0%, rgba(251,125,91,.10), transparent 60%), {BG}; }}
h1, h2, h3, h4 {{ font-family: 'Space Grotesk', 'Inter', sans-serif !important; letter-spacing: -.01em; }}
header[data-testid="stHeader"] {{ background: rgba(11,20,22,.75); backdrop-filter: blur(10px);
  border-bottom: 1px solid {LINE}; }}
.stDeployButton, [data-testid="stToolbarActions"], [data-testid="stSidebarCollapsedControl"] {{ display: none; }}
.block-container {{ padding-top: 4.6rem; padding-bottom: 3rem; max-width: 1240px; }}
@keyframes rise {{ from {{ opacity: 0; transform: translateY(10px); }} to {{ opacity: 1; transform: none; }} }}
@keyframes grow {{ from {{ width: 0; }} to {{ width: var(--w); }} }}
@keyframes ring {{ from {{ stroke-dashoffset: 327; }} }}
@keyframes pulse {{ 0%, 100% {{ box-shadow: 0 0 0 0 rgba(45,212,191,.45); }} 50% {{ box-shadow: 0 0 0 7px rgba(45,212,191,0); }} }}

/* hero */
.hero {{ padding: 6px 0 18px; animation: rise .5s ease both; }}
.hero .tag {{ display: inline-flex; align-items: center; gap: 8px; font-size: .78rem; font-weight: 600;
  color: {TEAL}; background: rgba(45,212,191,.10); border: 1px solid rgba(45,212,191,.25);
  padding: 5px 12px; border-radius: 999px; }}
.hero .tag i {{ width: 7px; height: 7px; border-radius: 50%; background: {TEAL}; animation: pulse 2s infinite; }}
.hero h1 {{ font-size: 2.6rem; line-height: 1.1; margin: 14px 0 8px; padding: 0;
  background: linear-gradient(90deg, #FFFFFF 0%, #B9F3EA 60%, {TEAL} 100%);
  -webkit-background-clip: text; background-clip: text; color: transparent; }}
.hero p {{ color: {MUTED}; font-size: 1.05rem; margin: 0; max-width: 720px; }}

/* glass panels */
[data-testid="stVerticalBlockBorderWrapper"] {{ background: rgba(18,33,38,.72); border: 1px solid {LINE} !important;
  border-radius: 18px; backdrop-filter: blur(6px); }}
.label {{ color: {MUTED}; font-size: .8rem; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; }}

/* summary strip */
.strip {{ display: grid; grid-template-columns: 1.4fr 1fr 1fr 1fr; gap: 14px; margin: 18px 0 8px; }}
.stat {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 16px 18px; animation: rise .5s ease both; }}
.stat .v {{ font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700; color: {TEXT}; line-height: 1.1; }}
.stat .k {{ color: {MUTED}; font-size: .85rem; margin-top: 2px; }}
.stat.big .v {{ font-size: 2.4rem; color: {TEAL}; }}
.dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 8px; }}

/* recommendation cards */
.picks {{ display: grid; grid-template-columns: 1.35fr 1fr 1fr; gap: 14px; margin: 6px 0 4px; }}
.pick {{ position: relative; background: {CARD}; border: 1px solid {LINE}; border-radius: 18px; padding: 20px;
  animation: rise .5s ease both; transition: transform .2s, border-color .2s; overflow: hidden; }}
.pick:hover {{ transform: translateY(-3px); border-color: var(--c); }}
.pick.first {{ background: linear-gradient(135deg, rgba(45,212,191,.16), rgba(18,33,38,.9) 60%);
  border-color: rgba(45,212,191,.45); }}
.pick .rank {{ font-size: .75rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: {CORAL}; }}
.pick .name {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 600; color: {TEXT}; margin: 4px 0 10px; }}
.pick .row {{ display: flex; align-items: center; gap: 18px; }}
.pick .free {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.6rem; font-weight: 700; color: {TEXT}; line-height: 1; }}
.pick .free small {{ display: block; font-family: 'Inter', sans-serif; font-size: .85rem; font-weight: 500; color: {MUTED}; margin-top: 4px; }}
.pick .meta {{ color: {MUTED}; font-size: .85rem; margin-top: 12px; }}
.pick .meta b {{ color: {TEXT}; }}
.ring {{ flex: none; }}
.ring circle.v {{ animation: ring 1.1s ease both; }}
.badge {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 11px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; color: var(--c); background: color-mix(in srgb, var(--c) 14%, transparent);
  border: 1px solid color-mix(in srgb, var(--c) 35%, transparent); }}

/* car park tiles */
.tiles {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; margin-top: 6px; }}
.tile {{ background: {CARD}; border: 1px solid {LINE}; border-left: 4px solid var(--c); border-radius: 14px;
  padding: 13px 14px; animation: rise .45s ease both; transition: transform .18s, box-shadow .18s; }}
.tile:hover {{ transform: translateY(-3px); box-shadow: 0 10px 28px -12px var(--c); }}
.tile .n {{ font-size: .8rem; color: {MUTED}; font-weight: 600; letter-spacing: .02em; }}
.tile .f {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; color: {TEXT}; margin: 4px 0 8px; }}
.tile .f small {{ font-family: 'Inter', sans-serif; font-size: .8rem; font-weight: 500; color: {MUTED}; }}
.bar {{ height: 7px; background: rgba(255,255,255,.07); border-radius: 99px; overflow: hidden; }}
.bar i {{ display: block; height: 100%; width: var(--w); border-radius: 99px; background: var(--c); animation: grow .9s ease both; }}
.tile .s {{ display: flex; justify-content: space-between; color: {MUTED}; font-size: .78rem; margin-top: 8px; }}
.tile .real {{ margin-top: 8px; padding-top: 8px; border-top: 1px dashed {LINE}; font-size: .78rem; color: {TEXT}; }}

/* day timeline */
.timeline {{ display: grid; grid-template-columns: repeat(18, 1fr); gap: 4px; margin: 8px 0 2px; }}
.timeline div {{ height: 42px; border-radius: 8px; background: var(--c); opacity: var(--o); animation: rise .4s ease both;
  display: flex; align-items: flex-end; justify-content: center; font-size: .66rem; color: #0B1416; font-weight: 700; padding-bottom: 4px; }}
.tl-axis {{ display: grid; grid-template-columns: repeat(18, 1fr); gap: 4px; color: {MUTED}; font-size: .7rem; text-align: center; }}

/* steps */
.steps {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }}
.step {{ background: {CARD}; border: 1px solid {LINE}; border-top: 3px solid var(--c); border-radius: 16px; padding: 16px;
  animation: rise .5s ease both; }}
.step .num {{ font-family: 'Space Grotesk', 'Inter', sans-serif; font-size: .8rem; font-weight: 700; color: var(--c); }}
.step .t {{ font-weight: 700; color: {TEXT}; margin: 4px 0 6px; }}
.step .x {{ color: {MUTED}; font-size: .84rem; line-height: 1.45; }}

.legend {{ display: flex; gap: 18px; flex-wrap: wrap; color: {MUTED}; font-size: .85rem; margin: 2px 0 6px; }}
.section {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 600; color: {TEXT}; margin: 26px 0 4px; }}
.sub {{ color: {MUTED}; font-size: .92rem; margin-bottom: 10px; }}
.note {{ background: rgba(96,165,250,.08); border: 1px solid rgba(96,165,250,.25); border-radius: 14px;
  padding: 12px 16px; color: {TEXT}; font-size: .9rem; margin-top: 14px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ background: {CARD}; border-radius: 10px; padding: 8px 16px; border: 1px solid {LINE}; }}
.stTabs [aria-selected="true"] {{ border-color: {TEAL}; }}
@media (max-width: 900px) {{ .strip, .picks {{ grid-template-columns: 1fr 1fr; }} .steps {{ grid-template-columns: 1fr 1fr; }}
  .hero h1 {{ font-size: 2rem; }} }}
/* answer card */
.answer {{ display: flex; align-items: center; justify-content: space-between; gap: 24px; margin: 18px 0 12px;
  padding: 26px 30px; border-radius: 22px; border: 1px solid color-mix(in srgb, var(--c) 45%, transparent);
  background: linear-gradient(120deg, color-mix(in srgb, var(--c) 16%, transparent), rgba(18,33,38,.92) 55%);
  animation: rise .5s ease both; }}
.a-name {{ font-family: 'Space Grotesk', 'Inter', sans-serif; font-size: 2.6rem; font-weight: 700; color: {TEXT}; line-height: 1.1; margin: 6px 0; }}
.a-name small {{ font-family: 'Inter', sans-serif; font-size: .85rem; font-weight: 500; color: {MUTED}; margin-left: 12px; }}
.a-free {{ font-size: 1.15rem; color: {MUTED}; display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
.a-free b {{ color: {TEXT}; font-size: 1.6rem; font-family: 'Space Grotesk', 'Inter', sans-serif; }}
.why {{ margin-top: 14px; color: {TEXT}; font-size: .98rem; }}
.why span {{ color: var(--c); font-weight: 700; }}
.trust {{ margin-top: 6px; color: {MUTED}; font-size: .88rem; }}
.tip {{ border-radius: 12px; padding: 12px 16px; font-size: .98rem; color: {TEXT}; margin-top: 4px; }}
.tip.ok {{ background: rgba(52,211,153,.08); border: 1px solid rgba(52,211,153,.3); }}
.tip.warn {{ background: rgba(251,191,36,.08); border: 1px solid rgba(251,191,36,.3); }}
iframe {{ border-radius: 14px; }}
.st-key-role_tabs {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 5px; margin-top: 8px; }}
.st-key-role_tabs button {{ border-radius: 10px !important; border: 0 !important; min-height: 44px; }}
.st-key-role_tabs button[kind="secondary"] {{ background: transparent !important; color: {MUTED} !important; }}
.st-key-role_tabs button[kind="secondary"]:hover {{ color: {TEXT} !important; background: rgba(255,255,255,.04) !important; }}
.st-key-role_tabs button[kind="primary"] {{ color: #06201D !important; font-weight: 600; }}
.auth-title {{ font-family: 'Space Grotesk', 'Inter', sans-serif; font-size: 1.75rem; font-weight: 700; color: {TEXT}; margin: 26px 0 2px; }}
.auth-sub {{ color: {MUTED}; font-size: .95rem; margin-bottom: 14px; }}
.st-key-auth_switch {{ display: flex; justify-content: center; }}
.st-key-auth_switch button {{ color: {MUTED} !important; }}
.st-key-auth_switch button strong {{ color: {TEAL}; }}
.or {{ display: flex; align-items: center; gap: 12px; color: {MUTED}; font-size: .8rem; margin: 2px 0 10px; }}
.or::before, .or::after {{ content: ""; flex: 1; border-top: 1px solid {LINE}; }}
@media (max-width: 700px) {{
  .st-key-role_tabs [data-testid="stHorizontalBlock"] {{ flex-wrap: nowrap !important; gap: 6px; }}
  .st-key-role_tabs [data-testid="stColumn"] {{ min-width: 0 !important; width: 50% !important; flex: 1 1 50% !important; }}
  .signin-art {{ display: none; }}
}}
[data-testid="stHeaderActionElements"], .stMarkdown h1 a, .stMarkdown h2 a {{ display: none !important; }}
.st-key-site_footer {{ margin-top: 20px; padding: 30px 32px 18px; border-radius: 22px 22px 0 0;
  background: linear-gradient(180deg, rgba(18,33,38,.95), rgba(11,20,22,1)); border: 1px solid {LINE}; border-bottom: 0; }}
.st-key-site_footer [data-testid="stPageLink"] a {{ padding: 2px 0; background: transparent !important; }}
.st-key-site_footer [data-testid="stPageLink"] a span, .st-key-site_footer [data-testid="stPageLink"] a p {{ font-weight: 400 !important; }}
.st-key-site_footer [data-testid="stPageLink"] p {{ color: {MUTED}; font-size: .9rem; }}
.st-key-site_footer [data-testid="stPageLink"] a:hover p {{ color: {TEAL}; }}
.f-brand {{ display: flex; align-items: center; gap: 10px; font-family: 'Space Grotesk', 'Inter', sans-serif; font-weight: 700;
  font-size: 1.1rem; color: {TEXT}; margin-bottom: 10px; }}
.f-brand .logo {{ width: 32px; height: 32px; border-radius: 9px; background: {TEAL}; color: #06201D; display: inline-flex;
  align-items: center; justify-content: center; font-weight: 800; }}
.f-text {{ color: {MUTED}; font-size: .88rem; line-height: 1.5; }}
.f-chip {{ display: inline-block; margin-top: 12px; padding: 4px 11px; border-radius: 999px; font-size: .76rem; font-weight: 600;
  color: {TEAL}; background: rgba(45,212,191,.10); border: 1px solid rgba(45,212,191,.25); }}
.f-head {{ color: {TEXT}; font-size: .8rem; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; margin-bottom: 8px; }}
.f-key {{ color: {MUTED}; font-size: .86rem; padding: 3px 0; }}
.f-key b {{ color: {TEXT}; font-weight: 600; }}
.f-bottom {{ display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-top: 22px; padding-top: 14px;
  border-top: 1px solid {LINE}; color: {MUTED}; font-size: .78rem; }}
.signin-art {{ background: linear-gradient(150deg, rgba(45,212,191,.18), rgba(18,33,38,.95) 55%, rgba(251,125,91,.12));
  border: 1px solid {LINE}; border-radius: 22px; padding: 28px 30px; animation: rise .5s ease both; }}
.signin-art .brand {{ display: flex; align-items: center; gap: 10px; font-family: 'Space Grotesk', 'Inter', sans-serif;
  font-weight: 700; font-size: 1.05rem; color: {TEXT}; }}
.signin-art .logo {{ width: 34px; height: 34px; border-radius: 10px; background: {TEAL}; color: #06201D; display: inline-flex;
  align-items: center; justify-content: center; font-weight: 800; }}
.signin-art h2 {{ font-size: 1.9rem; color: {TEXT}; margin: 22px 0 16px; line-height: 1.15; }}
.signin-art ul {{ list-style: none; padding: 0; margin: 18px 0 14px; }}
.signin-art li {{ color: {TEXT}; padding: 7px 0 7px 30px; position: relative; font-size: .98rem; }}
.signin-art li::before {{ content: "✓"; position: absolute; left: 0; top: 6px; width: 20px; height: 20px; border-radius: 50%;
  background: rgba(45,212,191,.18); color: {TEAL}; font-size: .75rem; font-weight: 800; display: flex; align-items: center; justify-content: center; }}
.signin-art .small {{ color: {MUTED}; font-size: .8rem; margin-top: 8px; }}
.demo {{ margin-top: 14px; padding: 11px 14px; border-radius: 12px; background: rgba(96,165,250,.08);
  border: 1px dashed rgba(96,165,250,.35); color: {TEXT}; font-size: .88rem; text-align: center; }}
.alert {{ display: flex; justify-content: space-between; gap: 12px; padding: 11px 14px; margin-bottom: 8px; border-radius: 12px;
  background: rgba(248,113,113,.08); border: 1px solid rgba(248,113,113,.3); color: {TEXT}; font-size: .92rem; animation: rise .4s ease both; }}
.alert.ok {{ background: rgba(52,211,153,.07); border-color: rgba(52,211,153,.28); }}
.alert small {{ color: {MUTED}; margin-left: 6px; }}
@media (max-width: 700px) {{ .picks {{ grid-template-columns: 1fr !important; }} .strip {{ grid-template-columns: 1fr 1fr !important; }} .answer {{ flex-direction: column-reverse; align-items: flex-start; }} .a-name {{ font-size: 2rem; }} }}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- small helpers
def hero(tag, title, text):
    st.markdown(f'<div class="hero"><span class="tag"><i></i>{tag}</span><h1>{title}</h1><p>{text}</p></div>',
                unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="section">{title}</div>' + (f'<div class="sub">{sub}</div>' if sub else ""),
                unsafe_allow_html=True)


def legend():
    return ('<div class="legend">' + "".join(
        f'<span><span class="dot" style="background:{c}"></span><b style="color:{TEXT}">{ICON[k]} {n}</b> · {d}</span>'
        for k, (n, c, d) in STATUS.items()) + "</div>")


def ring(pct, col, size=118):
    r, c = 52, 2 * np.pi * 52
    off = c * (1 - min(max(pct, 0), 1))
    return (f'<svg class="ring" width="{size}" height="{size}" viewBox="0 0 120 120">'
            f'<circle cx="60" cy="60" r="{r}" fill="none" stroke="rgba(255,255,255,.08)" stroke-width="11"/>'
            f'<circle class="v" cx="60" cy="60" r="{r}" fill="none" stroke="{col}" stroke-width="11" stroke-linecap="round" '
            f'stroke-dasharray="{c:.1f}" stroke-dashoffset="{off:.1f}" transform="rotate(-90 60 60)"/>'
            f'<text x="60" y="58" text-anchor="middle" fill="{TEXT}" font-family="Space Grotesk, Inter, sans-serif" font-size="24" font-weight="700">{pct:.0%}</text>'
            f'<text x="60" y="78" text-anchor="middle" fill="{MUTED}" font-family="Inter, sans-serif" font-size="11">full</text></svg>')


def dark_fig(fig, height=420):
    fig.update_layout(height=height, template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter, sans-serif", color=TEXT, size=13), margin=dict(t=30, b=10, l=10, r=10),
                      hoverlabel=dict(bgcolor=CARD, bordercolor=TEAL, font_family="Inter"))
    fig.update_xaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,.06)", zeroline=False)
    return fig


def day_label(d):
    return pd.Timestamp(d).strftime("%a %d %b")


# ----------------------------------------------------------------------------- loading
@st.cache_data
def load_grid():
    return pd.read_csv(C.GRID_PARQUET, parse_dates=["slot_time", "date"])


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
    df["status_id"] = rate_to_status(df.pred)
    df["free_spaces"] = ((1 - df.pred.clip(0, 1)) * df.capacity).round().astype(int)
    return df


def deployed(task):
    res = load_json("results.json")
    if not res:
        return None
    r = res[task]
    tm = pd.DataFrame(r["test_metrics"]).T
    dep, base = r["deployed_model"], r["best_baseline"]
    return dict(cars=tm.loc[dep, "MAE_vehicles"], acc=tm.loc[dep, "status_acc"],
                better=1 - tm.loc[dep, "MAE"] / tm.loc[base, "MAE"], table=tm, dep=dep, base=base)


# ----------------------------------------------------------------------------- friendly names and plain words
@st.cache_data
def friendly_names():
    """Car parks with a real name keep it; coded ones become 'Car Park 1', 'Car Park 2', ..."""
    zones = sorted(load_grid().zone.unique())
    names, n = {}, 0
    for z in zones:
        if any(ch.isdigit() for ch in z):
            n += 1
            names[z] = f"Car Park {n}"
        else:
            names[z] = z
    return names


def nice(zone):
    return friendly_names().get(zone, zone)


def trust_words(acc):
    return "about 19 times out of 20" if acc >= 0.93 else "about 9 times out of 10"


def why_line(zone, day, slot, task):
    """One plain sentence built from the clues the model leans on most (its usual level, last week, right now)."""
    g = load_grid()
    z = g[g.zone == zone]
    weekend = pd.Timestamp(day).dayofweek >= 5
    hist = z[(z.slot == slot) & (z.date <= pd.Timestamp(C.TRAIN_END)) & ((z.dow >= 5) == weekend)].rate
    usual = hist.mean()
    kind = "weekends" if weekend else "weekdays"
    if task == "short":
        now = z[(z.date == pd.Timestamp(day)) & (z.slot == slot - C.SHORT_HORIZON_SLOTS)].rate
        before = z[(z.date == pd.Timestamp(day)) & (z.slot == slot - C.SHORT_HORIZON_SLOTS - 1)].rate
        if len(now) and not np.isnan(now.iloc[0]):
            trend = ""
            if len(before) and not np.isnan(before.iloc[0]):
                d = now.iloc[0] - before.iloc[0]
                trend = " and filling up" if d > 0.01 else " and emptying" if d < -0.01 else " and steady"
            return (f"It is {now.iloc[0]:.0%} full right now ({SLOT_LABELS[slot - C.SHORT_HORIZON_SLOTS]}){trend}, "
                    f"and it is usually about {usual:.0%} full on {kind} at {SLOT_LABELS[slot]}.")
    last = z[(z.date == pd.Timestamp(day) - pd.Timedelta(days=7)) & (z.slot == slot)].rate
    lw = f", and it was {last.iloc[0]:.0%} full at this time last week" if len(last) and not np.isnan(last.iloc[0]) else ""
    return f"It is usually about {usual:.0%} full on {kind} at {SLOT_LABELS[slot]}{lw}."


ICON = {0: "✓", 1: "!", 2: "✕"}


def badge(sid, text=None):
    name, col, _ = STATUS[int(sid)]
    return f'<span class="badge" style="--c:{col}">{ICON[int(sid)]} {text or name}</span>'


def play_the_day(df, day, start_slot, task, staff=False, limit=None):
    """Animated grid of every car park for one day. Runs in the browser, so it is smooth:
    press play and watch the car parks fill up and empty from 08:00 to 16:30."""
    d = df[df.date.dt.date == day]
    slots = sorted(d.slot.unique())
    zones = []
    for z, gz in d.groupby("zone"):
        gz = gz.set_index("slot")
        zones.append({"n": nice(z), "c": z if nice(z) != z else "", "cap": int(gz.capacity.iloc[0]),
                      "p": [None if s not in gz.index else round(float(gz.pred[s]), 4) for s in slots],
                      "f": [None if s not in gz.index else int(gz.free_spaces[s]) for s in slots],
                      "r": [None if s not in gz.index or np.isnan(gz.rate[s]) else round(float(gz.rate[s]), 4) for s in slots]})
    data = {"labels": [SLOT_LABELS[s] for s in slots], "start": slots.index(start_slot) if start_slot in slots else 0,
            "zones": zones, "lim": C.LIMITED_THRESHOLD, "full": C.FULL_THRESHOLD,
            "mode": "forecast made the evening before" if task == "day_ahead" else "forecast made 1 hour before",
            "staff": staff, "limit": limit or len(zones)}
    if not staff:
        for z in data["zones"]:
            z["c"] = ""
    html = PLAY_HTML.replace("__DATA__", json.dumps(data))
    st.iframe(html, height="content")


PLAY_HTML = """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box} body{margin:0;background:transparent;font-family:Inter,system-ui,sans-serif;color:#E8F2F0}
.top{display:flex;align-items:center;gap:16px;flex-wrap:wrap;background:#122126;border:1px solid rgba(255,255,255,.08);
 border-radius:16px;padding:14px 18px;margin-bottom:14px}
button{background:#2DD4BF;color:#06201D;border:0;border-radius:999px;padding:10px 20px;font:700 15px Inter,sans-serif;cursor:pointer;
 display:flex;align-items:center;gap:8px;transition:transform .15s,box-shadow .15s}
button:hover{transform:scale(1.04);box-shadow:0 0 0 6px rgba(45,212,191,.15)}
.time{font:700 34px 'Space Grotesk',Inter,sans-serif;min-width:96px;font-variant-numeric:tabular-nums}
input[type=range]{flex:1;min-width:180px;accent-color:#2DD4BF}
.sum{display:flex;gap:16px;flex-wrap:wrap;font-size:14px;color:#8FA6A3;width:100%}
.sum b{color:#E8F2F0;font-variant-numeric:tabular-nums} .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
label.rv{font-size:14px;color:#8FA6A3;display:flex;align-items:center;gap:6px;cursor:pointer;margin-left:auto}
.grid{position:relative}
.tile{position:absolute;left:0;top:0;height:112px;background:#122126;border:1px solid rgba(255,255,255,.08);border-left:4px solid #34D399;
 border-radius:14px;padding:11px 13px;transition:transform .7s cubic-bezier(.2,.8,.2,1),border-color .5s,opacity .4s}
.tile:hover{box-shadow:0 10px 26px -12px #2DD4BF}
.nm{font-weight:600;font-size:14px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cd{font-size:11px;color:#8FA6A3;margin-left:6px;font-weight:500}
.fr{font:700 26px 'Space Grotesk',Inter,sans-serif;margin:3px 0 7px;font-variant-numeric:tabular-nums;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
@media (max-width:480px){.fr{font-size:22px}.fr small{font-size:11px}.top{gap:10px;padding:12px}.time{font-size:28px;min-width:80px}}
.fr small{font:500 12px Inter,sans-serif;color:#8FA6A3}
.bar{height:7px;background:rgba(255,255,255,.07);border-radius:99px;overflow:hidden;position:relative}
.bar i{display:block;height:100%;border-radius:99px;transition:width .7s ease,background .5s}
.bar u{position:absolute;top:-2px;width:3px;height:11px;background:#fff;border-radius:2px;transition:left .7s ease;display:none}
.st{display:flex;justify-content:space-between;font-size:12px;color:#8FA6A3;margin-top:7px;white-space:nowrap}
.rv-on .w{display:none} .rv-on .st{justify-content:flex-end}
.st span:first-child{font-weight:600}
.rv-on .bar u{display:block}
.real{display:none} .rv-on .real{display:inline}
.driver .rv, .driver .pc{display:none} .driver .sum .mode{display:none}
</style>
<div class="top">
  <button id="play">▶ Play the day</button>
  <div class="time" id="time">--:--</div>
  <input type="range" id="slider" min="0" step="1">
  <label class="rv"><input type="checkbox" id="rv"> Show what really happened <span style="color:#fff">(white mark)</span></label>
  <div class="sum" id="sum"></div>
</div>
<div class="grid" id="grid"></div>
<script>
const D = __DATA__;
const COL = ['#34D399', '#FBBF24', '#F87171'], WORD = ['✓ Available', '! Limited', '✕ Full'];
const st = p => p >= D.full ? 2 : p >= D.lim ? 1 : 0;
const grid = document.getElementById('grid'), slider = document.getElementById('slider');
slider.max = D.labels.length - 1;
let idx = D.start, timer = null, shown = {};
if (!D.staff) document.body.classList.add('driver');
const tiles = D.zones.map((z, k) => {
  const el = document.createElement('div'); el.className = 'tile';
  el.innerHTML = `<div class="nm">${z.n}<span class="cd">${z.c}</span></div><div class="fr"><span class="v">0</span> <small>free of ${z.cap.toLocaleString()}</small></div>` +
    `<div class="bar"><i></i><u></u></div><div class="st"><span class="w"></span><span><span class="pc"></span><span class="real"></span></span></div>`;
  grid.appendChild(el); shown[k] = 0; return el;
});
function tween(el, from, to) {
  const t0 = performance.now();
  (function f(t) { const a = Math.min(1, (t - t0) / 600), e = 1 - Math.pow(1 - a, 3);
    el.textContent = Math.round(from + (to - from) * e).toLocaleString(); if (a < 1) requestAnimationFrame(f); })(t0);
}
function layout(order) {
  const W = grid.clientWidth, cols = W >= 600 ? Math.floor((W + 12) / 212) : 2, w = (W - (cols - 1) * 12) / cols;
  order.forEach((k, i) => { const el = tiles[k]; el.style.width = w + 'px';
    el.style.transform = `translate(${(i % cols) * (w + 12)}px, ${Math.floor(i / cols) * 124}px)`; });
  grid.style.height = Math.ceil(order.length / cols) * 124 + 'px';
}
function show(i) {
  idx = i; slider.value = i; document.getElementById('time').textContent = D.labels[i];
  const live = D.zones.map((z, k) => k).filter(k => D.zones[k].p[i] !== null);
  D.zones.forEach((z, k) => { tiles[k].style.opacity = z.p[i] === null ? 0 : 1; });
  const free = k => D.zones[k].f[i];
  live.sort((a, b) => free(b) - free(a));
  let tot = 0, cnt = [0, 0, 0];
  live.forEach(k => { const z = D.zones[k], p = z.p[i], s = st(p), el = tiles[k], f = free(k); tot += f; cnt[s]++;
    el.style.borderLeftColor = COL[s];
    const bar = el.querySelector('.bar i'); bar.style.width = Math.min(100, p * 100) + '%'; bar.style.background = COL[s];
    el.querySelector('.w').textContent = WORD[s]; el.querySelector('.w').style.color = COL[s];
    el.querySelector('.pc').textContent = 'forecast ' + Math.round(p * 100) + '%';
    const r = z.r[i], u = el.querySelector('.bar u');
    el.querySelector('.real').textContent = r === null ? '' : ' · really ' + Math.round(r * 100) + '% full';
    u.style.left = r === null ? '-10px' : `calc(${Math.min(100, r * 100)}% - 1px)`;
    tween(el.querySelector('.v'), shown[k], f); shown[k] = f; });
  const hidden = live.slice(D.limit).concat(D.zones.map((z, k) => k).filter(k => D.zones[k].p[i] === null));
  hidden.forEach(k => { tiles[k].style.opacity = 0; tiles[k].style.pointerEvents = 'none'; tiles[k].style.transform = 'translate(0px, 0px)'; });
  live.slice(0, D.limit).forEach(k => { tiles[k].style.pointerEvents = 'auto'; });
  layout(live.slice(0, D.limit));
  grid.style.height = Math.ceil(Math.min(D.limit, live.length) / (grid.clientWidth >= 600 ? Math.floor((grid.clientWidth + 12) / 212) : 2)) * 124 + 'px';
  document.getElementById('sum').innerHTML = `<span><b>${tot.toLocaleString()}</b> free spaces in all car parks<span class="mode"> (${D.mode})</span></span>` +
    [0, 1, 2].map(s => `<span><span class="dot" style="background:${COL[s]}"></span><b>${cnt[s]}</b> ${WORD[s].slice(2).toLowerCase()}</span>`).join('');
  if (!D.staff) tiles.forEach(el => { el.querySelector('.fr small').textContent = 'free spaces'; });
}
function stop() { clearInterval(timer); timer = null; document.getElementById('play').textContent = '▶ Play the day'; }
document.getElementById('play').onclick = () => {
  if (timer) return stop();
  if (idx >= D.labels.length - 1) show(0);
  document.getElementById('play').textContent = '❚❚ Pause';
  timer = setInterval(() => { if (idx >= D.labels.length - 1) return stop(); show(idx + 1); }, 1100);
};
slider.oninput = () => { stop(); show(+slider.value); };
document.getElementById('rv').onchange = e => document.body.classList.toggle('rv-on', e.target.checked);
window.addEventListener('resize', () => show(idx));
show(idx);
</script>
"""


# ----------------------------------------------------------------------------- page 1: find parking
def page_find_parking():
    who = st.session_state.get("user") if st.session_state.get("role") == "driver" else None
    guest = st.session_state.get("role") == "guest"
    hero(f"Welcome back, {who.title()} · replaying real days from December 2016" if who else
         ("Browsing as a guest · replaying real days from December 2016" if guest else
          "Replaying real days from December 2016"), "Where should I park?",
         "Choose when you are arriving. We show you the car park most likely to have a free space.")
    fa = forecast_frame("day_ahead")
    dates = sorted(fa.date.dt.date.unique())

    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1.6, 1.1], gap="large")
        day = c1.selectbox("Arrival day", dates, index=0, format_func=day_label)
        slot = c2.select_slider("Arrival time", options=list(range(C.N_SLOTS)), value=12,
                                format_func=lambda i: SLOT_LABELS[i])
        mode = c3.segmented_control("How are you planning?", ["Planning ahead", "Leaving now"],
                                    default="Planning ahead",
                                    help="Planning ahead: the forecast was made the evening before.\n\n"
                                         "Leaving now: you arrive in about 1 hour, and the forecast also knows "
                                         "how full each car park is right now, so it is more accurate.") or "Planning ahead"

    task = "day_ahead" if mode == "Planning ahead" else "short"
    if task == "short" and slot < C.SHORT_HORIZON_SLOTS:
        st.info(f"\"Leaving now\" looks 1 hour ahead, so it starts from {SLOT_LABELS[C.SHORT_HORIZON_SLOTS]}. "
                f"Showing {SLOT_LABELS[C.SHORT_HORIZON_SLOTS]} instead.")
        slot = C.SHORT_HORIZON_SLOTS
    df = forecast_frame(task)
    sel = df[(df.date.dt.date == day) & (df.slot == slot)].sort_values("free_spaces", ascending=False)
    if sel.empty:
        st.info("There is no forecast for this time. Please pick another time.")
        return
    info = deployed(task)
    best = sel.iloc[0]
    col = STATUS[int(best.status_id)][1]
    trust = (f'Forecasts like this are right {trust_words(info["acc"])}.' if info else "")
    st.markdown(
        f'<div class="answer" style="--c:{col}">'
        f'<div class="a-left"><div class="label">Best place to park · {day_label(day)}, {SLOT_LABELS[slot]}</div>'
        f'<div class="a-name">{nice(best.zone)}</div>'
        f'<div class="a-free"><b>{best.free_spaces:,}</b> free spaces expected {badge(best.status_id)}</div>'
        f'<div class="why"><span>Why we think so:</span> {why_line(best.zone, day, slot, task)}</div>'
        f'<div class="trust">{trust}</div></div>'
        f'<div class="a-ring">{ring(best.pred, col, 150)}</div></div>', unsafe_allow_html=True)

    others = []
    for i, (_, r) in enumerate(sel.iloc[1:3].iterrows()):
        c = STATUS[int(r.status_id)][1]
        others.append(f'<div class="pick" style="--c:{c};animation-delay:{(i + 1) * 90}ms">'
                      f'<div class="rank">{["Also good", "Also good"][i]}</div>'
                      f'<div class="row">{ring(r.pred, c, 76)}<div><div class="name">{nice(r.zone)}</div>'
                      f'<div class="meta"><b>{r.free_spaces:,}</b> free spaces &nbsp;{badge(r.status_id)}</div></div></div></div>')
    st.markdown('<div class="picks" style="grid-template-columns:1fr 1fr">' + "".join(others) + "</div>",
                unsafe_allow_html=True)

    with st.container(border=True):
        names = {nice(z): z for z in sorted(sel.zone, key=lambda z: (not nice(z).startswith("Car Park"),
                                                                       int(nice(z).split()[-1]) if nice(z).startswith("Car Park") else 0, nice(z)))}
        options = ["No, any car park is fine"] + list(names)
        pick = st.selectbox("Going to a particular car park?", options,
                            help="We will tell you how full it will be, and suggest a better one if it is busy.")

        if pick != "No, any car park is fine":
            r = sel[sel.zone == names[pick]].iloc[0]
            sid = int(r.status_id)
            if sid == 0:
                st.markdown(f'<div class="tip ok">{badge(0)} &nbsp;<b>{pick}</b> should be fine: about '
                            f'<b>{r.free_spaces:,}</b> free spaces ({r.pred:.0%} full).</div>', unsafe_allow_html=True)
            else:
                alt = sel[(sel.status_id == 0) & (sel.zone != r.zone)].head(1)
                alt_txt = (f' Try <b>{nice(alt.iloc[0].zone)}</b> instead: about <b>{alt.iloc[0].free_spaces:,}</b> '
                           f'free spaces.' if len(alt) else " Every car park is busy at this time; try arriving earlier.")
                st.markdown(f'<div class="tip warn">{badge(sid)} &nbsp;<b>{pick}</b> will be {r.pred:.0%} full '
                            f'(only about {r.free_spaces:,} spaces left).{alt_txt}</div>', unsafe_allow_html=True)

    section("How busy will it get today?",
            "Press ▶ to watch the day from 8 am to 4:30 pm. Green = easy to park, yellow = filling up, red = full.")
    show_all = st.toggle("Show all car parks", value=False, help="Normally we show the 10 with the most free spaces.")
    play_the_day(df, day, slot, task, staff=False, limit=None if show_all else 10)

# ----------------------------------------------------------------------------- page 2: one car park
def page_zone_forecast():
    hero("Plan your visit", "When is the best time to go?",
         "Choose a car park and a day. See how full it gets hour by hour, and the best time to arrive.")
    df = forecast_frame("day_ahead")
    with st.container(border=True):
        c1, c2 = st.columns([1.3, 1], gap="large")
        zones = sorted(df.zone.unique())
        zones = sorted(zones, key=lambda z: (not nice(z).startswith("Car Park"), int(nice(z).split()[-1]) if nice(z).startswith("Car Park") else 0, nice(z)))
        zone = c1.selectbox("Car park", zones, index=zones.index("BHMBCCTHL01") if "BHMBCCTHL01" in zones else 0, format_func=nice)
        dates = sorted(df.date.dt.date.unique())
        day = c2.selectbox("Day", dates, format_func=day_label, key="zf_day")
    d = df[(df.zone == zone) & (df.date.dt.date == day)].sort_values("slot")
    if d.empty:
        st.info("No forecast for this car park on this day.")
        return

    cap = int(d.capacity.iloc[0])
    peak = d.loc[d.pred.idxmax()]
    busy_slots = d[d.status_id >= 1]
    first_busy = busy_slots.iloc[0] if len(busy_slots) else None
    full_slots = d[d.status_id == 2]
    cards = [
        ("Easy parking until", SLOT_LABELS[int(first_busy.slot)] if first_busy is not None else "All day",
         "arrive before this to find a space easily" if first_busy is not None else "plenty of space all day",
         "#34D399"),
        ("Busiest time", SLOT_LABELS[int(peak.slot)], f"{peak.pred:.0%} full · {int(peak.free_spaces):,} spaces left",
         STATUS[int(peak.status_id)][1]),
        ("Full for", f"{len(full_slots) * 30} min" if len(full_slots) else "Never",
         "expected to be 90% full or more" if len(full_slots) else "never expected to be full", CORAL),
        ("Size", f"{cap:,}", "parking spaces in total", SKY),
    ]
    st.markdown('<div class="strip" style="grid-template-columns:repeat(4,1fr)">' + "".join(
        f'<div class="stat" style="animation-delay:{i * 70}ms;border-top:3px solid {c}"><div class="k">{k}</div>'
        f'<div class="v" style="font-size:1.8rem;margin-top:4px">{v}</div><div class="k">{h}</div></div>'
        for i, (k, v, h, c) in enumerate(cards)) + "</div>", unsafe_allow_html=True)

    section("The day at a glance", f"{nice(zone)} · {day_label(day)} · {cap:,} spaces. Each block is half an hour.")
    blocks = []
    for i, (_, r) in enumerate(d.iterrows()):
        blocks.append(f'<div title="{SLOT_LABELS[int(r.slot)]}: {r.pred:.0%} full ({STATUS[int(r.status_id)][0]})" '
                      f'style="--c:{STATUS[int(r.status_id)][1]};--o:{0.45 + 0.55 * min(r.pred, 1):.2f};'
                      f'animation-delay:{i * 30}ms">{r.pred:.0%}</div>')
    st.markdown('<div class="timeline">' + "".join(blocks) + '</div><div class="tl-axis">'
                + "".join(f"<span>{SLOT_LABELS[int(i)] if int(i) % 2 == 0 else ''}</span>" for i in d.slot) + "</div>"
                + legend(), unsafe_allow_html=True)

    section("How full, hour by hour")
    st.plotly_chart(day_chart(d, None, False), width="stretch")


def day_chart(d, s, reveal):
    x = [SLOT_LABELS[i] for i in d.slot]
    fig = go.Figure()
    for y0, y1, sid in ((0, C.LIMITED_THRESHOLD, 0), (C.LIMITED_THRESHOLD, C.FULL_THRESHOLD, 1), (C.FULL_THRESHOLD, 1.05, 2)):
        fig.add_hrect(y0=y0, y1=y1, fillcolor=STATUS[sid][1], opacity=0.06, line_width=0, layer="below",
                      annotation_text=STATUS[sid][0], annotation_position="top left",
                      annotation_font=dict(color=STATUS[sid][1], size=12))
    fig.add_trace(go.Scatter(x=x, y=d.pred, name="Forecast (made the evening before)", mode="lines",
                             line=dict(color=TEAL, width=3.5, shape="spline"), fill="tozeroy",
                             fillcolor="rgba(45,212,191,0.12)", hovertemplate="%{x}: %{y:.0%} full<extra>Forecast</extra>"))
    if s is not None and len(s):
        fig.add_trace(go.Scatter(x=[SLOT_LABELS[i] for i in s.slot], y=s.pred, name="Forecast (1 hour before)",
                                 mode="lines", line=dict(color=CORAL, width=2.5, dash="dash", shape="spline"),
                                 hovertemplate="%{x}: %{y:.0%} full<extra>1 hour before</extra>"))
    if reveal:
        fig.add_trace(go.Scatter(x=x, y=d.rate, name="What really happened", mode="lines+markers",
                                 line=dict(color="#FFFFFF", width=2), marker=dict(size=7, color="#FFFFFF"),
                                 hovertemplate="%{x}: %{y:.0%} full<extra>Really</extra>"))
    fig.update_layout(yaxis=dict(title="How full", tickformat=".0%", range=[0, 1.05]),
                      legend=dict(orientation="h", y=-0.18), hovermode="x unified", showlegend=s is not None or reveal)
    return dark_fig(fig, 440)


# ----------------------------------------------------------------------------- page 3: staff dashboard
STAFF_HEROES = {
    "busy": ("Busy times", "When parking is busiest: across all car parks, and for each car park on a normal weekday."),
    "groups": ("Car park groups: busy and quiet", "The computer sorted the car parks into two groups by how their days look, without being told the answer."),
    "unusual": ("Unusual days", "Days that looked strange, flagged automatically so staff can check the sensors or the car park."),
    "accuracy": ("How accurate are the forecasts?", "Measured on the December days the computer had never seen."),
}


def page_admin(part):
    title, text = STAFF_HEROES[part]
    hero("For parking staff", title, text)
    grid = load_grid()
    g = grid.dropna(subset=["rate"])
    an = load_csv("zone_day_anomalies.csv")
    tab1, tab2, tab3, tab4 = (part == "busy", part == "groups", part == "unusual", part == "accuracy")
    if tab1:
        hm = g.pivot_table(index="dow", columns="slot", values="rate", aggfunc="mean")
        fig = go.Figure(go.Heatmap(z=hm.values, x=SLOT_LABELS, y=[DOW[i] for i in hm.index], colorscale=HEAT,
                                   zmin=0, zmax=0.9, xgap=3, ygap=3,
                                   colorbar=dict(tickformat=".0%", title="How full", outlinewidth=0),
                                   hovertemplate="%{y} %{x}: %{z:.0%} full<extra></extra>"))
        fig.update_layout(title="Busiest around 2 pm and on Thursdays, quietest on Sundays",
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(dark_fig(fig, 380), width="stretch")
        zh = g[g.dow < 5].pivot_table(index="zone", columns="slot", values="rate", aggfunc="mean")
        zh = zh.loc[zh.mean(axis=1).sort_values().index]
        fig = go.Figure(go.Heatmap(z=zh.values, x=SLOT_LABELS, y=[nice(z) for z in zh.index], colorscale=HEAT, zmin=0, zmax=1, xgap=2, ygap=2,
                                   colorbar=dict(tickformat=".0%", title="How full", outlinewidth=0),
                                   hovertemplate="%{y} %{x}: %{z:.0%} full<extra></extra>"))
        fig.update_layout(title="Each car park on a normal weekday (busiest at the top)")
        st.plotly_chart(dark_fig(fig, 700), width="stretch")
    if tab2:
        cl = load_csv("zone_clusters.csv")
        cl["mean"] = cl.zone.map(g[g.dow < 5].groupby("zone").rate.mean())
        cl["capacity"] = cl.zone.map(grid.groupby("zone").capacity.first())
        busy = cl.groupby("cluster")["mean"].mean().idxmax()
        cl["group"] = np.where(cl.cluster == busy, "Busy", "Quiet")
        a, b = st.columns(2)
        for col, grp, colour in ((a, "Busy", CORAL), (b, "Quiet", TEAL)):
            sub = cl[cl.group == grp]
            col.markdown(f'<div class="stat" style="border-top:3px solid {colour}"><div class="k">{grp} car parks</div>'
                         f'<div class="v">{len(sub)}</div><div class="k">on a normal weekday, on average {sub["mean"].mean():.0%} full</div></div>',
                         unsafe_allow_html=True)
        c = cl.sort_values("mean")
        fig = go.Figure()
        for grp, colour in (("Quiet", TEAL), ("Busy", CORAL)):
            sub = c[c.group == grp]
            fig.add_trace(go.Bar(x=sub["mean"], y=[nice(z) for z in sub.zone], orientation="h", name=grp, marker_color=colour,
                                 customdata=sub.capacity,
                                 hovertemplate="<b>%{y}</b><br>%{x:.0%} full on average<br>%{customdata:,} spaces<extra></extra>"))
        fig.update_xaxes(tickformat=".0%", title="Average fullness on a normal weekday")
        fig.update_layout(barmode="overlay", legend=dict(orientation="h", y=1.04), bargap=0.25)
        st.plotly_chart(dark_fig(fig, 720), width="stretch")
    if tab3:
        flagged = an[an.anomaly].sort_values("anomaly_score", ascending=False)
        st.markdown(f'<div class="sub">An automatic check looked at all {len(an):,} car-park days and flagged '
                    f'<b style="color:{TEXT}">{len(flagged)}</b> that looked strange, for example a sensor stuck on '
                    f'the same number or a car park that closed early. They are shown to staff, not deleted.</div>',
                    unsafe_allow_html=True)
        per = flagged.zone.value_counts().rename_axis("zone").reset_index(name="days").sort_values("days")
        fig = go.Figure(go.Bar(x=per.days, y=[nice(z) for z in per.zone], orientation="h", marker_color=VIOLET,
                               hovertemplate="<b>%{y}</b>: %{x} unusual days<extra></extra>"))
        fig.update_layout(title="Unusual days per car park")
        fig.update_xaxes(dtick=1, title="Unusual days")
        st.plotly_chart(dark_fig(fig, max(280, 24 * len(per) + 100)), width="stretch")
        table = flagged.assign(zone=[nice(z) for z in flagged.zone], date=pd.to_datetime(flagged.date).dt.strftime("%a %d %b %Y"),
                               mean=(flagged["mean"] * 100).round(), max=(flagged["max"] * 100).round())[
            ["zone", "date", "mean", "max", "stuck"]].rename(columns={
                "zone": "Car park", "date": "Day", "mean": "Average fullness", "max": "Highest fullness",
                "stuck": "Times the sensor repeated itself"})
        st.dataframe(table, hide_index=True, width="stretch", column_config={
            "Average fullness": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100),
            "Highest fullness": st.column_config.ProgressColumn(format="%d%%", min_value=0, max_value=100)})
    if tab4:
        task = st.segmented_control("Forecast", ["Evening before", "1 hour before"], default="Evening before",
                                    key="acc_task") or "Evening before"
        info = deployed("day_ahead" if task == "Evening before" else "short")
        if info is None:
            st.info("Results file not found.")
            return
        stats = [("Average difference", f"±{info['cars']:.0f} cars", "between forecast and reality", TEAL),
                 ("Better than a simple guess", f"{info['better']:.1%}", "than \"same as last week\"", CORAL),
                 ("Right label", f"{info['acc']:.0%}", f"right {trust_words(info['acc'])}", "#34D399")]
        st.markdown('<div class="strip" style="grid-template-columns:repeat(3,1fr)">' + "".join(
            f'<div class="stat" style="border-top:3px solid {c}"><div class="k">{k}</div>'
            f'<div class="v" style="margin-top:4px">{v}</div><div class="k">{h}</div></div>'
            for k, v, h, c in stats) + "</div>", unsafe_allow_html=True)
        names = {"Persistence (last value)": "Guess: same as now", "Persistence (yesterday)": "Guess: same as yesterday",
                 "Seasonal naive (last week)": "Guess: same as last week",
                 "Historical profile": "Guess: the usual level", "Stacked ensemble": "Team of models",
                 "SVR (RBF)": "SVR", "HistGB": "Gradient Boosting"}
        t = info["table"].sort_values("MAE_vehicles", ascending=False).reset_index().rename(columns={"index": "model"})
        t["label"] = [names.get(m, m) for m in t.model]
        colours = [TEAL if m == info["dep"] else ("rgba(251,125,91,.55)" if m in names and names[m].startswith("Guess")
                                                  else "rgba(96,165,250,.45)") for m in t.model]
        fig = go.Figure(go.Bar(x=t.MAE_vehicles, y=t.label, orientation="h", marker_color=colours,
                               text=[f"{v:.0f} cars" for v in t.MAE_vehicles], textposition="outside", cliponaxis=False,
                               hovertemplate="<b>%{y}</b><br>off by about %{x:.0f} cars on average<extra></extra>"))
        fig.update_layout(title="Average difference from reality (shorter is better)")
        fig.update_xaxes(title="cars")
        st.plotly_chart(dark_fig(fig, 430), width="stretch")
        st.markdown(f'<div class="legend"><span><span class="dot" style="background:{TEAL}"></span>'
                    f'Used in this app ({info["dep"]})</span><span><span class="dot" style="background:{SKY}"></span>'
                    f'Other computer models</span><span><span class="dot" style="background:{CORAL}"></span>'
                    f'Simple guesses</span></div>', unsafe_allow_html=True)
        section("Forecast vs reality for one car park", "Pick a car park and a day to compare the forecasts with what really happened.")
        df, sh = forecast_frame("day_ahead"), forecast_frame("short")
        c1, c2 = st.columns(2)
        zones = sorted(df.zone.unique(), key=nice)
        zone = c1.selectbox("Car park", zones, index=zones.index("BHMBCCTHL01") if "BHMBCCTHL01" in zones else 0,
                            format_func=lambda z: f"{nice(z)} ({z})" if nice(z) != z else z, key="acc_zone")
        day = c2.selectbox("Day", sorted(df.date.dt.date.unique()), format_func=day_label, key="acc_day")
        d = df[(df.zone == zone) & (df.date.dt.date == day)].sort_values("slot")
        s1 = sh[(sh.zone == zone) & (sh.date.dt.date == day)].sort_values("slot")
        if len(d):
            ok = d.dropna(subset=["rate"])
            if len(ok):
                st.markdown(f'<div class="sub">On this day the evening-before forecast was off by about '
                            f'<b style="color:{TEXT}">{np.mean(np.abs(ok.rate - ok.pred)) * d.capacity.iloc[0]:.0f} cars</b> on average.</div>',
                            unsafe_allow_html=True)
            st.plotly_chart(day_chart(d, s1, True), width="stretch")



# ----------------------------------------------------------------------------- staff: today's overview
def page_staff_overview():
    hero("For parking staff", "Today's overview",
         "Which car parks will be full, where to send drivers, and how the forecast compares with reality.")
    fa = forecast_frame("day_ahead")
    dates = sorted(fa.date.dt.date.unique())
    with st.container(border=True):
        c1, c2, c3 = st.columns([1, 1.6, 1.1], gap="large")
        day = c1.selectbox("Day", dates, index=0, format_func=day_label, key="so_day")
        slot = c2.select_slider("Time", options=list(range(C.N_SLOTS)), value=12, format_func=lambda i: SLOT_LABELS[i], key="so_slot")
        mode = c3.segmented_control("Forecast", ["Evening before", "1 hour before"], default="Evening before", key="so_mode") or "Evening before"
    task = "day_ahead" if mode == "Evening before" else "short"
    if task == "short" and slot < C.SHORT_HORIZON_SLOTS:
        slot = C.SHORT_HORIZON_SLOTS
    df = forecast_frame(task)
    sel = df[(df.date.dt.date == day) & (df.slot == slot)].sort_values("pred", ascending=False)
    an = load_csv("zone_day_anomalies.csv")
    odd = an[(an.anomaly) & (pd.to_datetime(an.date).dt.date == day)]
    counts = sel.status_id.value_counts()
    stats = [("Free spaces expected", f"{int(sel.free_spaces.sum()):,}", f"at {SLOT_LABELS[slot]} in all car parks", TEAL),
             ("Expected to be full", f"{int(counts.get(2, 0))}", "car parks 90% full or more", "#F87171"),
             ("Filling up", f"{int(counts.get(1, 0))}", "car parks 70% to 90% full", "#FBBF24"),
             ("Unusual readings", f"{len(odd)}", "car parks flagged on this day", VIOLET)]
    st.markdown('<div class="strip" style="grid-template-columns:repeat(4,1fr)">' + "".join(
        f'<div class="stat" style="animation-delay:{i * 70}ms;border-top:3px solid {c}"><div class="k">{k}</div>'
        f'<div class="v" style="margin-top:4px">{v}</div><div class="k">{h}</div></div>'
        for i, (k, v, h, c) in enumerate(stats)) + "</div>", unsafe_allow_html=True)

    full = sel[sel.status_id == 2]
    free = sel[sel.status_id == 0].sort_values("free_spaces", ascending=False)
    a, b = st.columns(2, gap="medium")
    with a:
        section("Needs attention", f"Expected to be full at {SLOT_LABELS[slot]}.")
        if full.empty:
            st.markdown('<div class="tip ok">No car park is expected to be full at this time.</div>', unsafe_allow_html=True)
        else:
            st.markdown("".join(
                f'<div class="alert"><span><b>{nice(r.zone)}</b> <small>{r.zone if nice(r.zone) != r.zone else ""}</small></span>'
                f'<span>{r.pred:.0%} full · {r.free_spaces:,} left</span></div>' for _, r in full.iterrows()),
                unsafe_allow_html=True)
    with b:
        section("Send drivers here", "Most free spaces at the same time.")
        st.markdown("".join(
            f'<div class="alert ok"><span><b>{nice(r.zone)}</b> <small>{r.zone if nice(r.zone) != r.zone else ""}</small></span>'
            f'<span>{r.free_spaces:,} free · {r.pred:.0%} full</span></div>' for _, r in free.head(5).iterrows()),
            unsafe_allow_html=True)
    if len(odd):
        st.markdown('<div class="note">Unusual readings on this day: ' + ", ".join(
            f"<b>{nice(z)}</b>" for z in odd.zone) + '. Check these sensors before trusting their numbers.</div>',
            unsafe_allow_html=True)

    section("The whole day, every car park",
            "Press ▶ to replay the day. Tick \"Show what really happened\" to see the real fullness as a white mark on each bar.")
    play_the_day(df, day, slot, task, staff=True)


# ----------------------------------------------------------------------------- page 4: how it works
def page_explain():
    hero("Behind the scenes", "How the forecast works",
         "The computer studied 77 days of real readings from 28 car parks, learned the patterns, "
         "and was then tested on days it had never seen.")
    steps = [("Collect", "35,717 readings of how many cars were in each car park, every 30 minutes.", SKY),
             ("Clean", "Removed repeated and impossible readings, and lined everything up by the half hour.", TEAL),
             ("Look for patterns", "Busy hours, busy days, and car parks that behave alike.", CORAL),
             ("Learn", "Tried several methods. The best was a Random Forest: hundreds of decision trees that vote.", VIOLET),
             ("Test fairly", "Learned from Oct–Nov, then tested on December days it had never seen.", "#FBBF24"),
             ("Use it", "This app turns each forecast into Available, Limited or Full.", "#34D399")]
    st.markdown('<div class="steps">' + "".join(
        f'<div class="step" style="--c:{c};animation-delay:{i * 70}ms"><div class="num">STEP {i + 1}</div>'
        f'<div class="t">{t}</div><div class="x">{x}</div></div>' for i, (t, x, c) in enumerate(steps)) + "</div>",
        unsafe_allow_html=True)

    section("How good is it?", "Measured on the December days the computer had never seen.")
    rows = []
    for task, label, sub in (("short", "Leaving now", "forecast 1 hour ahead"),
                             ("day_ahead", "Planning ahead", "forecast the evening before")):
        info = deployed(task)
        if info:
            rows.append(
                f'<div class="pick" style="--c:{TEAL}"><div class="rank">{label}</div><div class="name">{sub}</div>'
                f'<div class="row">{ring(info["acc"], "#34D399", 104).replace(">full<", ">right<")}'
                f'<div class="free">±{info["cars"]:.0f}<small>cars off, on average</small></div></div>'
                f'<div class="meta">{info["better"]:.1%} better than guessing \"same as last week\" · '
                f'right {trust_words(info["acc"])}</div></div>')
    st.markdown('<div class="picks" style="grid-template-columns:1fr 1fr">' + "".join(rows) + "</div>",
                unsafe_allow_html=True)

    section("What the computer pays most attention to")
    shap = load_json("shap_results.json")
    if shap:
        task = st.segmented_control("Forecast", ["Planning ahead", "Leaving now"], default="Planning ahead",
                                    key="ex_task") or "Planning ahead"
        m = shap["day_ahead" if task == "Planning ahead" else "short"]["mean_abs_shap"]
        top = sorted(m.items(), key=lambda kv: kv[1], reverse=True)[:6][::-1]
        vals = np.array([v for _, v in top])
        fig = go.Figure(go.Bar(x=vals / vals.max(), y=[CLUE_WORDS.get(k, k) for k, _ in top], orientation="h",
                               marker=dict(color=vals / vals.max(), colorscale=[[0, "#0F766E"], [1, TEAL]]),
                               hovertemplate="%{y}<extra></extra>"))
        fig.update_xaxes(showticklabels=False, title="importance (longer = matters more)", range=[0, 1.05])
        st.plotly_chart(dark_fig(fig, 330), width="stretch")
        st.caption("Measured with SHAP, a method that shows how much each clue pushed the forecast up or down.")
    st.markdown('<div class="note">The data comes from real public car parks in Birmingham, UK (October to December '
                '2016), used as a stand-in for a campus. The same code works with any campus parking records.</div>',
                unsafe_allow_html=True)


# ----------------------------------------------------------------------------- sign in (drivers optional, staff required)
# Drivers can always continue as a guest; signing in remembers their usual car park.
# Staff pages need a staff account. Passwords are stored as salted SHA-256 hashes, never as plain text.
# A real deployment would set its own accounts in Streamlit "secrets":
#   [accounts]  name = ["<sha256 of salt+password>", "driver" | "staff"]
SALT = "parking-intelligence"
DEMO = {"driver": ("drive2026", "driver"), "staff": ("parking2026", "staff")}


def _hash(password):
    return hashlib.sha256((SALT + password).encode()).hexdigest()


USERS_FILE = ROOT / "data" / "users.json"   # accounts people create in the app (hashed passwords only)


def _load_users():
    try:
        return {k: tuple(v) for k, v in json.loads(USERS_FILE.read_text()).items()}
    except Exception:
        return {}


def accounts():
    acc = {u: (_hash(pw), role) for u, (pw, role) in DEMO.items()}
    try:
        acc.update({k.lower(): tuple(v) for k, v in dict(st.secrets["accounts"]).items()})
    except Exception:
        pass
    acc.update(_load_users())
    return acc


def create_account(username, password, role):
    users = _load_users()
    users[username] = (_hash(password), role)
    USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users))


ROLE_TEXT = {
    "Driver": ("Find a free space before you arrive",
               ["See the best car park for your arrival time", "Check any car park, with a reason for every forecast",
                "Get a better option when it's going to be full"]),
    "Parking staff": ("Manage parking with confidence",
                      ["See which car parks will be full, and where to send drivers",
                       "Busy times, car park groups and unusual sensor readings",
                       "Check how accurate the forecasts are"]),
}
SIGNIN_ART = """<svg viewBox="0 0 300 120" width="100%" height="120" aria-hidden="true">
<rect x="0" y="0" width="300" height="120" rx="14" fill="rgba(255,255,255,.04)"/>
<g stroke="rgba(255,255,255,.18)" stroke-width="2">
<line x1="30" y1="20" x2="30" y2="100"/><line x1="78" y1="20" x2="78" y2="100"/><line x1="126" y1="20" x2="126" y2="100"/>
<line x1="174" y1="20" x2="174" y2="100"/><line x1="222" y1="20" x2="222" y2="100"/><line x1="270" y1="20" x2="270" y2="100"/></g>
<rect x="38" y="34" width="32" height="54" rx="8" fill="#F87171"/><rect x="86" y="34" width="32" height="54" rx="8" fill="#FBBF24"/>
<rect x="182" y="34" width="32" height="54" rx="8" fill="#F87171"/><rect x="230" y="34" width="32" height="54" rx="8" fill="#FBBF24"/>
<rect x="134" y="28" width="32" height="64" rx="8" fill="none" stroke="#2DD4BF" stroke-width="2.5" stroke-dasharray="6 5"/>
<text x="150" y="66" text-anchor="middle" font-family="Inter, sans-serif" font-size="22" font-weight="700" fill="#2DD4BF">P</text>
</svg>"""


def page_sign_in(default_role="Driver"):
    left, right = st.columns([1, 1.05], gap="large")
    with right:
        role = st.session_state.get("signin_role", default_role)
        mode = st.session_state.get("signin_mode", "Sign in")
        with st.container(key="role_tabs"):
            t1, t2 = st.columns(2, gap="small")
            for col, name, icon in ((t1, "Driver", ":material/directions_car:"), (t2, "Parking staff", ":material/badge:")):
                if col.button(name, icon=icon, width="stretch", key=f"tab_{name}_{default_role}",
                              type="primary" if role == name else "secondary"):
                    st.session_state["signin_role"] = name
                    st.rerun()
        want = "driver" if role == "Driver" else "staff"
        who = "driver" if want == "driver" else "parking staff"
        if mode == "Sign in":
            st.markdown(f'<div class="auth-title">Welcome back</div><div class="auth-sub">Sign in to your {who} account.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="auth-title">Create your account</div><div class="auth-sub">&nbsp;</div>',
                        unsafe_allow_html=True)
        if mode == "Sign in":
            with st.form(f"signin_{default_role}", border=False):
                user = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                ok = st.form_submit_button("Sign in", type="primary", width="stretch")
            if ok:
                u = user.strip().lower()
                acc = accounts().get(u)
                if acc and hmac.compare_digest(acc[0], _hash(password)) and acc[1] == want:
                    st.session_state.update(user=u, role=want, go_after_login=want)
                    st.rerun()
                elif acc and hmac.compare_digest(acc[0], _hash(password)):
                    st.error(f"This is a {'driver' if acc[1] == 'driver' else 'parking staff'} account. "
                             f"Choose \"{'Driver' if acc[1] == 'driver' else 'Parking staff'}\" at the top.")
                else:
                    st.error("That username or password isn't right. New here? Create an account below.")
        else:
            with st.form(f"signup_{default_role}", border=False):
                user = st.text_input("Choose a username", placeholder="At least 3 letters or numbers")
                password = st.text_input("Choose a password", type="password", placeholder="At least 6 characters")
                again = st.text_input("Type the password again", type="password", placeholder="Same password")
                ok = st.form_submit_button(f"Create {'driver' if want == 'driver' else 'staff'} account",
                                           type="primary", width="stretch")
            if ok:
                u = user.strip().lower()
                if len(u) < 3 or not u.replace("_", "").isalnum():
                    st.error("Use at least 3 letters or numbers for the username (no spaces).")
                elif u in accounts():
                    st.error("That username is already taken. Please choose another one.")
                elif len(password) < 6:
                    st.error("The password needs at least 6 characters.")
                elif password != again:
                    st.error("The two passwords don't match.")
                else:
                    create_account(u, password, want)
                    st.session_state.update(user=u, role=want, go_after_login=want)
                    st.rerun()
        with st.container(key="auth_switch"):
            if mode == "Sign in":
                if st.button("New here? **Create an account**", type="tertiary", key=f"to_signup_{default_role}"):
                    st.session_state["signin_mode"] = "Create account"
                    st.rerun()
            else:
                if st.button("Already have an account? **Sign in**", type="tertiary", key=f"to_signin_{default_role}"):
                    st.session_state["signin_mode"] = "Sign in"
                    st.rerun()
        if role == "Driver":
            st.markdown('<div class="or"><span>or</span></div>', unsafe_allow_html=True)
            if st.button("Continue as guest →", width="stretch"):
                st.session_state.update(role="guest", go_after_login="driver")
                st.rerun()
    title, points = ROLE_TEXT[role]
    with left:
        st.markdown(
            '<div class="signin-art"><div class="brand"><span class="logo">P</span>Parking Intelligence</div>'
            f'<h2>{title}</h2>' + SIGNIN_ART +
            '<ul>' + "".join(f"<li>{p}</li>" for p in points) + '</ul>'
            '<div class="small">Data: real car parks in Birmingham, UK (Oct–Dec 2016), used as a stand-in for a campus.</div></div>',
            unsafe_allow_html=True)


def page_sign_in_driver():
    page_sign_in("Driver")


def page_sign_out():
    for k in ("user", "role", "fav_car_park", "signin_mode"):
        st.session_state.pop(k, None)
    st.rerun()


def staff_busy(): page_admin("busy")
def staff_groups(): page_admin("groups")
def staff_unusual(): page_admin("unusual")
def staff_accuracy(): page_admin("accuracy")


# The menu depends on who is using the app:
#   nobody yet -> only the sign-in screen (choose: sign in, or continue as guest)
#   guest  -> Find parking · Plan your visit · Sign in
#   driver -> Find parking · Plan your visit · Sign out
#   staff  -> the five staff pages · Sign out
# "How it works" is linked from the footer of every page instead of the menu.
role = st.session_state.get("role")
FIND = st.Page(page_find_parking, title="Find parking", icon=":material/local_parking:", default=role in ("guest", "driver"))
PLAN = st.Page(page_zone_forecast, title="Plan your visit", icon=":material/schedule:", url_path="plan")
HOW = st.Page(page_explain, title="How it works", icon=":material/lightbulb:", url_path="how-it-works", visibility="hidden")
SIGN_IN = st.Page(page_sign_in_driver, title="Sign in", icon=":material/login:", url_path="sign-in", default=role is None)
SIGN_OUT = st.Page(page_sign_out, title=f"Sign out ({st.session_state.get('user', '').title()})",
                   icon=":material/logout:", url_path="sign-out")
STAFF_PAGES = [
    st.Page(page_staff_overview, title="Today's overview", icon=":material/dashboard:", url_path="staff", default=role == "staff"),
    st.Page(staff_busy, title="Busy times", icon=":material/local_fire_department:", url_path="busy-times"),
    st.Page(staff_groups, title="Car park groups", icon=":material/category:", url_path="groups"),
    st.Page(staff_unusual, title="Unusual days", icon=":material/warning:", url_path="unusual-days"),
    st.Page(staff_accuracy, title="Forecast accuracy", icon=":material/target:", url_path="accuracy")]

if role == "staff":
    pages = STAFF_PAGES + [SIGN_OUT, HOW]
elif role == "driver":
    pages = [FIND, PLAN, SIGN_OUT, HOW]
elif role == "guest":
    pages = [FIND, PLAN, SIGN_IN, HOW]
else:
    pages = [SIGN_IN, HOW]
nav = st.navigation(pages, position="top" if role else "hidden")
after = st.session_state.pop("go_after_login", None)
if after == "staff":
    st.switch_page(STAFF_PAGES[0])
elif after == "driver":
    st.switch_page(FIND)
nav.run()

# ----------------------------------------------------------------------------- footer
if role == "staff":
    quick = [(p, None) for p in STAFF_PAGES[:3]]
elif role in ("guest", "driver"):
    quick = [(FIND, None), (PLAN, None), (SIGN_OUT if role == "driver" else SIGN_IN, None)]
else:
    quick = [(SIGN_IN, None)]
with st.container(key="site_footer"):
    c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1.2], gap="large")
    with c1:
        st.markdown('<div class="f-brand"><span class="logo">P</span>Parking Intelligence</div>'
                    '<div class="f-text">Predicts how full each car park will be, so drivers find a free space '
                    'faster and parking staff can plan ahead.</div>'
                    '<div class="f-chip">Replaying real days · 5–19 December 2016</div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="f-head">Quick links</div>', unsafe_allow_html=True)
        for pg, _ in quick:
            st.page_link(pg)
    with c3:
        st.markdown('<div class="f-head">About</div>', unsafe_allow_html=True)
        st.page_link(HOW, label="How it works", icon=":material/lightbulb:")
        st.page_link("https://github.com/TanyaAgrahari9/Parking-Intelligence", label="Project code", icon=":material/code:")
        st.page_link("https://archive.ics.uci.edu/dataset/482/parking+birmingham", label="Data source (UCI)",
                     icon=":material/database:")
    with c4:
        st.markdown('<div class="f-head">What the colours mean</div>'
                    '<div class="f-key"><span class="dot" style="background:#34D399"></span><b>✓ Available</b> · under 70% full</div>'
                    '<div class="f-key"><span class="dot" style="background:#FBBF24"></span><b>! Limited</b> · 70% to 90% full</div>'
                    '<div class="f-key"><span class="dot" style="background:#F87171"></span><b>✕ Full</b> · 90% full or more</div>'
                    '<div class="f-text" style="margin-top:8px">Forecasts are right about 9 times out of 10.</div>',
                    unsafe_allow_html=True)
    st.markdown('<div class="f-bottom"><span>© 2026 Tanya Agrahari · CSE513 Machine Learning project · '
                'Lovely Professional University</span><span>Data: Parking Birmingham, UCI Machine Learning Repository</span></div>',
                unsafe_allow_html=True)
