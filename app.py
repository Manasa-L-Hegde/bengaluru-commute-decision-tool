"""
app.py - Bengaluru Commute Decision Tool Dashboard
Run: streamlit run app.py
"""
import os, sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from pathlib import Path
from datetime import date, timedelta

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from weather_api import get_weather

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru Commute Intelligence",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-header {
    background: linear-gradient(135deg, #1A1D2E 0%, #0D1B4B 50%, #1A1D2E 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #6C63FF, #FF6584, #43E97B);
}
.hero-title {
    font-size: 2rem; font-weight: 700; color: #E8EAF6; margin: 0;
    background: linear-gradient(135deg, #E8EAF6, #6C63FF);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #9FA8DA; font-size: 0.95rem; margin-top: 6px; }

.card {
    background: linear-gradient(145deg, #1A1D2E, #141728);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.card:hover { border-color: rgba(108,99,255,0.6); }
.card-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #7986CB; margin-bottom: 6px;
}
.metric-big {
    font-size: 2.4rem; font-weight: 700; color: #E8EAF6; line-height: 1;
}
.metric-sub { font-size: 0.85rem; color: #9FA8DA; margin-top: 4px; }

.badge {
    display: inline-block; padding: 4px 14px; border-radius: 20px;
    font-size: 0.8rem; font-weight: 600; letter-spacing: 0.05em;
}
.badge-low      { background:#1B4D3E; color:#43E97B; border:1px solid #43E97B44; }
.badge-moderate { background:#3D2E00; color:#FFD54F; border:1px solid #FFD54F44; }
.badge-high     { background:#4A1500; color:#FF8A65; border:1px solid #FF856544; }
.badge-critical { background:#3B0A0A; color:#FF5252; border:1px solid #FF525244; }

/* Dual-mode source badges */
.mode-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 5px 16px; border-radius: 24px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em;
}
.mode-historical {
    background: rgba(66,165,245,0.12); color: #42A5F5;
    border: 1px solid rgba(66,165,245,0.35);
}
.mode-predicted {
    background: rgba(255,183,77,0.12); color: #FFB74D;
    border: 1px solid rgba(255,183,77,0.35);
}

.alt-route-card {
    background: linear-gradient(145deg, #0D2137, #0A1929);
    border: 1px solid rgba(67,233,123,0.3);
    border-radius: 14px; padding: 20px 24px; margin-top: 8px;
}
.alt-route-title { font-size:0.75rem; font-weight:600; color:#43E97B;
    letter-spacing:0.1em; text-transform:uppercase; margin-bottom:8px; }
.alt-route-name  { font-size:1.4rem; font-weight:700; color:#E8EAF6; }
.alt-route-meta  { font-size:0.85rem; color:#80CBC4; margin-top:4px; }

.section-header {
    font-size: 1rem; font-weight: 600; color: #9FA8DA;
    letter-spacing: 0.08em; text-transform: uppercase;
    border-left: 3px solid #6C63FF; padding-left: 12px;
    margin: 28px 0 16px;
}
.source-badge {
    display:inline-block; font-size:0.68rem; background:#1A237E22;
    border:1px solid #3F51B544; color:#7986CB; border-radius:8px;
    padding:2px 10px; float:right; margin-top:2px;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data"
DASH = DATA / "dashboard"

SLOT_HOUR = {
    "Morning Peak (07-09)": 8,
    "Midday (11-13)":       12,
    "Evening Peak (17-19)": 18,
    "Night (21-23)":        22,
}

@st.cache_data(show_spinner=False)
def load_all():
    df   = pd.read_csv(DATA / "processed.csv", parse_dates=["date"])
    df["hour"] = df["time_slot"].map(SLOT_HOUR)
    q1   = pd.read_csv(DASH / "q1_hour_x_route.csv")
    q2   = pd.read_csv(DASH / "q2_hourly_trend.csv")
    q3   = pd.read_csv(DASH / "q3_route_heatmap.csv")
    return df, q1, q2, q3

@st.cache_resource(show_spinner=False)
def load_model():
    """Load the trained RandomForest model bundle."""
    model_path = ROOT / "models" / "risk_model.pkl"
    if model_path.exists():
        return joblib.load(model_path)
    return None

def _try_bq(query: str):
    """Attempt a BigQuery query; return None if unavailable."""
    try:
        project = os.getenv("GCP_PROJECT", "")
        if not project:
            return None
        from google.cloud import bigquery
        client = bigquery.Client(project=project)
        return client.query(query).to_dataframe()
    except Exception:
        return None


def _has_bigquery_credentials() -> bool:
    """Return True only when BigQuery access is likely available."""
    project = os.getenv("GCP_PROJECT", "")
    if not project:
        return False

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    if cred_path:
        return Path(cred_path).expanduser().exists()

    return False

# Historical data boundary
HIST_DATE_MAX = date(2024, 8, 9)

with st.spinner("Loading data…"):
    df, q1, q2, q3 = load_all()
    model_bundle = load_model()

routes = sorted(df["route"].unique())
areas  = sorted(df["area"].unique())
slots  = list(SLOT_HOUR.keys())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚦 Commute Planner")
    st.markdown("---")

    sel_area = st.selectbox("📍 Area", ["All Areas"] + areas)
    route_options = (
        sorted(df[df["area"] == sel_area]["route"].unique())
        if sel_area != "All Areas" else routes
    )
    sel_route = st.selectbox("🛣️ Route", route_options)
    sel_slot  = st.selectbox("⏱️ Time Slot", slots, index=0)
    sel_date  = st.date_input(
        "📅 Date",
        value=date.today(),
        min_value=date(2022, 1, 1),
        max_value=date.today() + timedelta(days=16),
    )

    st.markdown("---")
    st.markdown("##### 🌦️ Weather Filter")
    sel_weather = st.multiselect(
        "Conditions",
        sorted(df["weather_conditions"].unique()),
        default=[],
        placeholder="All conditions",
    )

    st.markdown("---")
    st.caption("Data: Jan 2022 – Aug 2024")
    st.caption("Weather: Open-Meteo API")
    bq_status = "🟢 BigQuery" if _has_bigquery_credentials() else "🟡 Offline (CSV)"
    st.caption(f"Source: {bq_status}")

# ── Dual-mode: Historical vs Predicted ────────────────────────────────────────
if sel_date is None:
    st.info("Please select a date to continue.")
    st.stop()

sel_hour = SLOT_HOUR[sel_slot]
sel_date_str = sel_date.strftime("%Y-%m-%d")
is_historical = (date(2022, 1, 1) <= sel_date <= HIST_DATE_MAX)
data_mode = "historical" if is_historical else "predicted"

def _predict_for_route(route_name, slot_name, target_date, rain_mm):
    """Use the ML model to predict risk_score and TTI for a route."""
    if model_bundle is None:
        return None
    le_route = model_bundle["le_route"]
    le_slot  = model_bundle["le_slot"]
    features = model_bundle["features"]
    rf_risk  = model_bundle["risk_model"]
    rf_tti   = model_bundle["tti_model"]

    # Encode route & slot
    if route_name not in le_route.classes_ or slot_name not in le_slot.classes_:
        return None
    route_enc = le_route.transform([route_name])[0]
    slot_enc  = le_slot.transform([slot_name])[0]
    dow = target_date.weekday()

    # Use historical averages for the route as baseline features
    rdf = df[df["route"] == route_name]
    cong_avg  = rdf["congestion_level"].mean() if not rdf.empty else 50.0
    cap_avg   = rdf["road_capacity_utilization"].mean() if "road_capacity_utilization" in rdf.columns and not rdf.empty else 70.0
    inc_avg   = rdf["incident_reports"].mean() if "incident_reports" in rdf.columns and not rdf.empty else 1.0
    vis_avg   = rdf["min_visibility_m"].mean() if "min_visibility_m" in rdf.columns and not rdf.empty else 8000.0
    ws_avg    = rdf["weather_severity"].mean() if "weather_severity" in rdf.columns and not rdf.empty else 1.0

    feat_vals = {
        "route_enc": route_enc, "day_of_week": dow, "time_slot_enc": slot_enc,
        "total_rain_mm": rain_mm, "congestion_level": cong_avg,
        "road_capacity_utilization": cap_avg, "incident_reports": inc_avg,
        "min_visibility_m": vis_avg, "weather_severity": ws_avg,
    }
    X_pred = pd.DataFrame([{f: feat_vals.get(f, 0) for f in features}])
    pred_risk = rf_risk.predict(X_pred)[0]
    pred_tti  = rf_tti.predict(X_pred)[0]
    return {"risk_score": pred_risk, "tti": pred_tti, "congestion": cong_avg}


route_q1 = q1[(q1["route"] == sel_route) & (q1["hour"] == sel_hour)]
route_df  = df[
    (df["route"] == sel_route) &
    (df["hour"]  == sel_hour)
]
if sel_weather:
    route_df = route_df[route_df["weather_conditions"].isin(sel_weather)]

if route_df.empty:
    route_df = df[(df["route"] == sel_route)]

if is_historical:
    # ── Historical mode: use actual data ──
    risk_score  = route_q1["avg_risk_score"].values[0]   if not route_q1.empty else route_df["risk_score"].mean()
    tti         = route_q1["avg_travel_time_index"].values[0] if not route_q1.empty else route_df["travel_time_index"].mean()
    cong        = route_q1["avg_congestion_level"].values[0]  if not route_q1.empty else route_df["congestion_level"].mean()
    speed       = route_q1["avg_speed_kmh"].values[0]    if not route_q1.empty else route_df["average_speed"].mean()
    rain        = route_q1["avg_rain_mm"].values[0]       if not route_q1.empty else route_df["total_rain_mm"].mean()
    tier        = route_q1["dominant_risk_tier"].values[0] if not route_q1.empty else route_df["risk_tier"].mode().iloc[0]
else:
    # ── Predicted mode: use ML model + live weather ──
    weather_info = get_weather(sel_date, sel_slot, df)
    rain = weather_info["rain_mm"]
    pred = _predict_for_route(sel_route, sel_slot, sel_date, rain)
    if pred:
        risk_score = pred["risk_score"]
        tti        = pred["tti"]
        cong       = pred["congestion"]
    else:
        # Fallback to historical averages
        risk_score = route_df["risk_score"].mean()
        tti        = route_df["travel_time_index"].mean()
        cong       = route_df["congestion_level"].mean()
    speed = route_df["average_speed"].mean()  # baseline from history
    # Derive tier from predicted risk score
    if risk_score >= 75:   tier = "Critical"
    elif risk_score >= 50: tier = "High"
    elif risk_score >= 25: tier = "Moderate"
    else:                  tier = "Low"

# Alternate route – lowest risk at same hour, different route
if is_historical:
    alt_candidates = q1[
        (q1["hour"]  == sel_hour) &
        (q1["route"] != sel_route)
    ].sort_values("avg_risk_score")
    alt_row  = alt_candidates.iloc[0] if not alt_candidates.empty else None
    alt_route = alt_row["route"]    if alt_row is not None else "N/A"
    alt_risk  = alt_row["avg_risk_score"] if alt_row is not None else 0
    alt_tti   = alt_row["avg_travel_time_index"] if alt_row is not None else 0
    alt_area  = alt_row["area"] if alt_row is not None else ""
else:
    # Predict for all other routes and pick the best
    alt_results = []
    for r in routes:
        if r == sel_route:
            continue
        p = _predict_for_route(r, sel_slot, sel_date, rain)
        if p:
            r_area = df[df["route"] == r]["area"].iloc[0] if not df[df["route"] == r].empty else ""
            alt_results.append({"route": r, "risk": p["risk_score"], "tti": p["tti"], "area": r_area})
    if alt_results:
        alt_results.sort(key=lambda x: x["risk"])
        best = alt_results[0]
        alt_route, alt_risk, alt_tti, alt_area = best["route"], best["risk"], best["tti"], best["area"]
    else:
        alt_route, alt_risk, alt_tti, alt_area = "N/A", 0, 0, ""

risk_saving = round(risk_score - alt_risk, 1)

# ── Hero header ───────────────────────────────────────────────────────────────
mode_icon = "📂" if is_historical else "🤖"
mode_label = "Historical Data" if is_historical else "ML Predicted"
mode_cls   = "mode-historical" if is_historical else "mode-predicted"

st.markdown(f"""
<div class="hero-header">
  <p class="hero-title">🚦 Bengaluru Commute Intelligence</p>
  <p class="hero-sub">
    Real-time risk analysis · {sel_date.strftime('%A, %d %b %Y')} · {sel_slot}
    &nbsp;&nbsp;
    <span class="mode-badge {mode_cls}">{mode_icon} {mode_label}</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ── Recommendation card ───────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Route Recommendation</div>', unsafe_allow_html=True)

badge_cls = {
    "Low": "badge-low", "Moderate": "badge-moderate",
    "High": "badge-high", "Critical": "badge-critical",
}.get(str(tier), "badge-moderate")

col_gauge, col_metrics, col_alt = st.columns([1.1, 1.4, 1.5], gap="medium")

with col_gauge:
    # Gauge chart
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(risk_score, 1),
        number={"font": {"size": 38, "color": "#E8EAF6"}, "suffix": ""},
        title={"text": "Risk Score", "font": {"size": 13, "color": "#9FA8DA"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#3F4460", "tickwidth": 1,
                     "tickfont": {"color": "#7986CB", "size": 10}},
            "bar": {"color": (
                "#FF5252" if risk_score >= 75 else
                "#FF8A65" if risk_score >= 50 else
                "#FFD54F" if risk_score >= 25 else "#43E97B"
            ), "thickness": 0.22},
            "bgcolor": "#1A1D2E",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25],  "color": "rgba(27,77,62,0.13)"},
                {"range": [25, 50], "color": "rgba(61,46,0,0.13)"},
                {"range": [50, 75], "color": "rgba(74,21,0,0.13)"},
                {"range": [75, 100],"color": "rgba(59,10,10,0.13)"},
            ],
            "threshold": {"line": {"color": "#6C63FF", "width": 3},
                          "thickness": 0.8, "value": risk_score},
        },
    ))
    gauge.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font_color="#E8EAF6",
    )
    st.plotly_chart(gauge, use_container_width=True)
    st.markdown(f"""
    <div style="text-align:center;margin-top:-10px">
      <span class="badge {badge_cls}">{tier} Risk</span>
      &nbsp;
      <span class="mode-badge {mode_cls}" style="font-size:0.68rem;padding:3px 10px">{mode_icon} {mode_label}</span>
    </div>""", unsafe_allow_html=True)

with col_metrics:
    st.markdown(f"""
    <div class="card">
      <div class="card-label">Selected Route</div>
      <div style="font-size:1.35rem;font-weight:700;color:#E8EAF6">{sel_route}</div>
      <div class="metric-sub">{sel_area if sel_area != 'All Areas' else q1[q1['route']==sel_route]['area'].iloc[0] if not q1[q1['route']==sel_route].empty else ''}</div>
    </div>
    <div class="card">
      <div class="card-label">Travel Time Index</div>
      <div class="metric-big">{tti:.3f}</div>
      <div class="metric-sub">×  vs free-flow speed (1.0 = no delay)</div>
    </div>
    <div class="card">
      <div class="card-label">Avg Speed · Congestion</div>
      <div class="metric-big">{speed:.1f} <span style="font-size:1rem;color:#9FA8DA">km/h</span></div>
      <div class="metric-sub">Congestion level: {cong:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col_alt:
    st.markdown(f"""
    <div class="card">
      <div class="card-label">Rainfall · Visibility</div>
      <div class="metric-big">{rain:.2f} <span style="font-size:1rem;color:#9FA8DA">mm</span></div>
      <div class="metric-sub">Peak-hour accumulated rain (avg over dataset)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alt-route-card">
      <div class="alt-route-title">💡 Recommended Alternate Route</div>
      <div class="alt-route-name">{alt_route}</div>
      <div class="alt-route-meta">{alt_area} &nbsp;·&nbsp; Risk: <b>{alt_risk:.1f}</b> &nbsp;·&nbsp; TTI: <b>{alt_tti:.3f}</b></div>
      <div style="margin-top:12px;font-size:0.82rem;color:#43E97B">
        ▼ {risk_saving} pts lower risk than selected route
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Trends ────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Trends <span class="source-badge">BigQuery / CSV</span></div>', unsafe_allow_html=True)

t1, t2 = st.columns(2, gap="medium")

with t1:
    st.markdown("**Hourly Risk & Travel Time — All Routes**")
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=q2["hour"], y=q2["avg_risk_score"],
        name="Avg Risk Score", mode="lines+markers",
        line=dict(color="#6C63FF", width=3),
        marker=dict(size=9, symbol="circle"),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.08)",
    ))
    fig_trend.add_trace(go.Scatter(
        x=q2["hour"], y=q2["max_risk_score"],
        name="Max Risk", mode="lines",
        line=dict(color="#FF5252", width=1.5, dash="dot"),
    ))
    fig_trend.add_trace(go.Scatter(
        x=q2["hour"], y=q2["min_risk_score"],
        name="Min Risk", mode="lines",
        line=dict(color="#43E97B", width=1.5, dash="dot"),
    ))
    fig_trend.update_layout(
        height=310, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9FA8DA", size=11),
        xaxis=dict(title="Hour", gridcolor="#1F2340", tickvals=[8,12,18,22],
                   ticktext=["07–09","11–13","17–19","21–23"]),
        yaxis=dict(title="Risk Score", gridcolor="#1F2340"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        margin=dict(l=10,r=10,t=10,b=10),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with t2:
    st.markdown("**Route Risk Comparison — Selected Hour**")
    hour_data = q1[q1["hour"] == sel_hour].sort_values("avg_risk_score", ascending=True)
    colors = [
        "#FF5252" if r >= 75 else "#FF8A65" if r >= 50 else "#FFD54F" if r >= 25 else "#43E97B"
        for r in hour_data["avg_risk_score"]
    ]
    fig_bar = go.Figure(go.Bar(
        x=hour_data["avg_risk_score"],
        y=hour_data["route"],
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:.1f}" for v in hour_data["avg_risk_score"]],
        textposition="outside", textfont=dict(color="#E8EAF6", size=10),
        hovertemplate="<b>%{y}</b><br>Risk: %{x:.1f}<extra></extra>",
    ))
    fig_bar.update_layout(
        height=310, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9FA8DA", size=10),
        xaxis=dict(title="Avg Risk Score", gridcolor="#1F2340", range=[0, 105]),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0)"),
        margin=dict(l=10,r=10,t=10,b=10),
        showlegend=False,
    )
    # Highlight selected route
    sel_idx = hour_data["route"].tolist().index(sel_route) if sel_route in hour_data["route"].values else -1
    if sel_idx >= 0:
        fig_bar.add_shape(type="rect",
            xref="paper", yref="y",
            x0=0, x1=1, y0=sel_idx - 0.4, y1=sel_idx + 0.4,
            fillcolor="rgba(108,99,255,0.15)", line=dict(color="#6C63FF", width=1),
            layer="below",
        )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Risk tier heatmap ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Risk Tier Heatmap — All Routes</div>', unsafe_allow_html=True)

tier_order = ["Low", "Moderate", "High", "Critical"]
tier_pivot = (
    q3.pivot_table(index="route", columns="risk_tier", values="pct_in_tier", fill_value=0)
      .reindex(columns=[t for t in tier_order if t in q3["risk_tier"].unique()], fill_value=0)
)
tier_colors = ["#1B4D3E", "#3D2E00", "#4A1500", "#3B0A0A"]
text_colors = ["#43E97B", "#FFD54F", "#FF8A65", "#FF5252"]

fig_heat = go.Figure()
for i, (col, fc, tc) in enumerate(zip(tier_pivot.columns, tier_colors, text_colors)):
    vals = tier_pivot[col].round(1)
    fig_heat.add_trace(go.Bar(
        name=col,
        x=tier_pivot.index,
        y=vals,
        marker_color=tc,
        marker_opacity=0.85,
        text=[f"{v}%" for v in vals],
        textposition="inside",
        textfont=dict(size=9, color="#E8EAF6"),
        hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.1f}}%<extra></extra>",
    ))

fig_heat.update_layout(
    barmode="stack", height=360,
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#9FA8DA", size=10),
    xaxis=dict(title="", tickangle=-35, gridcolor="rgba(0,0,0,0)"),
    yaxis=dict(title="% of Records", gridcolor="#1F2340"),
    legend=dict(orientation="h", y=1.05, bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=10, r=10, t=40, b=80),
)
st.plotly_chart(fig_heat, use_container_width=True)

# ── Raw data expander ─────────────────────────────────────────────────────────
with st.expander("🔍 Raw data for selected route / slot"):
    show_df = df[
        (df["route"] == sel_route) & (df["hour"] == sel_hour)
    ][["date","route","area","congestion_level","travel_time_index",
       "risk_score","risk_tier","weather_conditions","total_rain_mm","average_speed"]
    ].sort_values("date", ascending=False)
    st.dataframe(show_df, use_container_width=True, height=280)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#3F4460;font-size:0.78rem;margin-top:40px;padding-top:16px;
border-top:1px solid #1F2340">
  Bengaluru Commute Intelligence · Data: 2022–2024 · Weather: Open-Meteo · BigQuery Analytics
</div>
""", unsafe_allow_html=True)
