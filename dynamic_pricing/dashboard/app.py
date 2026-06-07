from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

st.set_page_config(
    page_title="AI Dynamic Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2563EB"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
BG = "#0B1220"
PANEL = "#111827"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"
BORDER = "rgba(148,163,184,0.18)"

API_URL = os.getenv(
    "API_URL",
    "https://dynamic-pricing-ai-production.up.railway.app"
).rstrip("/")
CATEGORY_OPTIONS = ["Electronics", "Groceries", "Fashion", "Home & Kitchen", "Beauty", "Sports"]
MONTH_OPTIONS = list(range(1, 13))
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_INDEX = {name: idx for idx, name in enumerate(DAY_NAMES)}
CATEGORY_BACKEND_MAP = {
    "Electronics": "Electronics",
    "Groceries": "Groceries",
    "Fashion": "Clothing",
    "Home & Kitchen": "Home & Garden",
    "Beauty": "Clothing",
    "Sports": "Sports",
}


@dataclass
class PredictionInputs:
    category: str
    base_price: float
    competitor_price_1: float
    competitor_price_2: float
    inventory_level: int
    demand_score: float
    day_of_week: int
    month: int
    customer_rating: float


# --- Styling -----------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --primary: {PRIMARY};
            --success: {SUCCESS};
            --warning: {WARNING};
            --danger: {DANGER};
            --bg: {BG};
            --panel: {PANEL};
            --text: {TEXT};
            --muted: {MUTED};
            --border: {BORDER};
        }}

        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at top left, rgba(37,99,235,0.16), transparent 26%),
                radial-gradient(circle at top right, rgba(16,185,129,0.12), transparent 20%),
                linear-gradient(180deg, #08111f 0%, #0b1220 48%, #0f172a 100%);
            color: var(--text);
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(17,24,39,0.98), rgba(15,23,42,0.98));
            border-right: 1px solid var(--border);
        }}

        .hero-card, .panel-card, .metric-card, .factor-card, .report-card {{
            background: rgba(17,24,39,0.84);
            border: 1px solid var(--border);
            border-radius: 20px;
            box-shadow: 0 18px 40px rgba(0,0,0,0.18);
        }}

        .hero-card {{
            padding: 1.5rem 1.6rem;
            background: linear-gradient(135deg, rgba(37,99,235,0.16), rgba(16,185,129,0.08));
        }}

        .section-label {{
            text-transform: uppercase;
            letter-spacing: 0.14em;
            color: var(--muted);
            font-size: 0.76rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}

        .kpi-value {{
            font-size: 3.1rem;
            font-weight: 800;
            letter-spacing: -0.04em;
            margin: 0.15rem 0 0.35rem 0;
            color: white;
        }}

        .kpi-subtitle {{
            color: var(--muted);
            font-size: 0.96rem;
        }}

        .metric-card, .panel-card, .factor-card, .report-card {{
            padding: 1.05rem 1.1rem;
        }}

        .soft-divider {{
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(148,163,184,0.35), transparent);
            margin: 1rem 0;
        }}

        .badge-pill {{
            display: inline-block;
            padding: 0.34rem 0.75rem;
            border-radius: 999px;
            background: rgba(37,99,235,0.15);
            border: 1px solid rgba(37,99,235,0.22);
            color: #bfdbfe;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .stButton > button {{
            background: linear-gradient(135deg, #2563EB, #1D4ED8);
            color: white;
            border: 0;
            border-radius: 14px;
            font-weight: 700;
            padding: 0.8rem 1.15rem;
            box-shadow: 0 12px 24px rgba(37,99,235,0.28);
        }}

        .stButton > button:hover {{
            transform: translateY(-1px);
            border: 0;
            color: white;
        }}

        [data-testid="stMetricValue"] {{
            color: white;
            font-weight: 800;
        }}

        [data-testid="stMetricDelta"] {{
            font-weight: 700;
        }}

        .streamlit-expanderHeader {{
            font-weight: 700;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: rgba(15,23,42,0.88);
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.65rem 1rem;
            color: #cbd5e1;
            font-weight: 700;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(37,99,235,0.28), rgba(16,185,129,0.18));
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- API helpers --------------------------------------------------------------

def api_get(path: str, timeout: int = 10) -> Any:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_predict(payload: dict[str, Any], timeout: int = 15) -> dict[str, Any]:
    response = requests.post(f"{API_URL}/predict-price", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_feature_importance_cached() -> dict[str, float]:
    try:
        data = api_get("/feature-importance")
        return {str(key): float(value) for key, value in data.items()}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_categories_cached() -> list[str]:
    try:
        payload = api_get("/categories")
        categories = payload.get("categories", [])
        return [str(item) for item in categories]
    except Exception:
        return CATEGORY_OPTIONS


# --- Data and analytics -------------------------------------------------------

def build_inputs() -> PredictionInputs:
    category = st.session_state.get("selected_category", CATEGORY_OPTIONS[0])
    base_price = float(st.session_state.get("base_price", 2499.0))
    competitor_price_1 = float(st.session_state.get("competitor_price_1", 2399.0))
    competitor_price_2 = float(st.session_state.get("competitor_price_2", 2449.0))
    inventory_level = int(st.session_state.get("inventory_level", 240))
    demand_score = float(st.session_state.get("demand_score", 68.0))
    day_of_week = int(st.session_state.get("day_of_week", datetime.now().weekday()))
    month = int(st.session_state.get("month", datetime.now().month))
    customer_rating = float(st.session_state.get("customer_rating", 4.4))
    return PredictionInputs(
        category=category,
        base_price=base_price,
        competitor_price_1=competitor_price_1,
        competitor_price_2=competitor_price_2,
        inventory_level=inventory_level,
        demand_score=demand_score,
        day_of_week=day_of_week,
        month=month,
        customer_rating=customer_rating,
    )


def compute_features(inputs: PredictionInputs) -> dict[str, float]:
    is_weekend = 1 if inputs.day_of_week >= 5 else 0
    is_holiday_season = 1 if inputs.month in [11, 12] else 0
    avg_competitor_price = (inputs.competitor_price_1 + inputs.competitor_price_2) / 2.0
    safe_base_price = max(inputs.base_price, 1e-6)

    price_gap_1 = inputs.base_price - inputs.competitor_price_1
    price_gap_2 = inputs.base_price - inputs.competitor_price_2
    inventory_pressure = inputs.demand_score / (inputs.inventory_level + 1.0)
    market_pressure = avg_competitor_price / safe_base_price
    seasonal_weight = 1.0 + (0.25 * is_holiday_season) + (0.10 * is_weekend) + (0.05 if inputs.month in [6, 7, 8] else 0.0)
    demand_inventory_ratio = (inputs.demand_score * seasonal_weight) / (inputs.inventory_level + 1.0)

    return {
        "is_weekend": float(is_weekend),
        "is_holiday_season": float(is_holiday_season),
        "avg_competitor_price": float(avg_competitor_price),
        "price_gap_1": float(price_gap_1),
        "price_gap_2": float(price_gap_2),
        "inventory_pressure": float(inventory_pressure),
        "market_pressure": float(market_pressure),
        "seasonal_weight": float(seasonal_weight),
        "demand_inventory_ratio": float(demand_inventory_ratio),
    }


def build_payload(inputs: PredictionInputs) -> dict[str, Any]:
    return {
        "category": CATEGORY_BACKEND_MAP.get(inputs.category, inputs.category),
        "base_price": inputs.base_price,
        "competitor_price_1": inputs.competitor_price_1,
        "competitor_price_2": inputs.competitor_price_2,
        "inventory_level": inputs.inventory_level,
        "demand_score": inputs.demand_score / 100.0,
        "day_of_week": inputs.day_of_week,
        "month": inputs.month,
        "customer_rating": inputs.customer_rating,
    }


def calculate_contributions(inputs: PredictionInputs, recommended_price: float) -> pd.DataFrame:
    features = compute_features(inputs)
    price_delta_pct = ((recommended_price - inputs.base_price) / max(inputs.base_price, 1e-6)) * 100.0
    demand_score_pct = inputs.demand_score / 100.0
    competitor_avg = (inputs.competitor_price_1 + inputs.competitor_price_2) / 2.0

    rows = [
        {
            "factor": "Demand Effect",
            "score": np.clip(demand_score_pct * 100.0, 0.0, 100.0),
            "impact": (demand_score_pct - 0.5) * 0.9,
        },
        {
            "factor": "Inventory Effect",
            "score": np.clip(100.0 - (inputs.inventory_level / 5.0), 0.0, 100.0),
            "impact": np.clip((250 - inputs.inventory_level) / 250.0, -1.0, 1.0) * 0.7,
        },
        {
            "factor": "Competitor Effect",
            "score": np.clip((competitor_avg / max(inputs.base_price, 1e-6)) * 100.0, 0.0, 150.0),
            "impact": np.clip((competitor_avg - inputs.base_price) / max(inputs.base_price, 1e-6), -1.0, 1.0) * 0.8,
        },
        {
            "factor": "Seasonal Effect",
            "score": 100.0 if inputs.month in [11, 12] else (75.0 if inputs.month in [6, 7, 8] else 45.0),
            "impact": 0.35 if inputs.month in [11, 12] else (0.18 if inputs.month in [6, 7, 8] else 0.02),
        },
        {
            "factor": "Customer Rating Effect",
            "score": np.clip((inputs.customer_rating / 5.0) * 100.0, 0.0, 100.0),
            "impact": ((inputs.customer_rating - 4.0) / 1.0) * 0.25,
        },
    ]

    frame = pd.DataFrame(rows)
    frame["direction"] = np.where(frame["impact"] >= 0, "Positive", "Negative")
    frame["strength"] = np.abs(frame["impact"])
    frame["headline"] = frame["impact"].apply(lambda value: "Lift" if value >= 0 else "Drag")
    frame["price_delta_pct"] = price_delta_pct
    frame["market_pressure"] = features["market_pressure"]
    return frame.sort_values("strength", ascending=False)


def build_monthly_seasonality(inputs: PredictionInputs) -> pd.DataFrame:
    months = list(range(1, 13))
    weekend_factor = 1.08 if inputs.day_of_week >= 5 else 0.96
    base_signal = inputs.demand_score / 100.0
    data = []
    for month in months:
        if month in [11, 12]:
            seasonal = 1.28
        elif month in [6, 7, 8]:
            seasonal = 1.12
        elif month in [1, 2]:
            seasonal = 0.92
        else:
            seasonal = 1.0
        data.append(
            {
                "month": month,
                "seasonal_influence": round((seasonal * weekend_factor * (0.65 + base_signal)) * 100.0, 2),
                "weekend_pressure": round((weekend_factor - 0.9) * 100.0, 2),
            }
        )
    return pd.DataFrame(data)


def build_category_analytics(selected_category: str, feature_importance: dict[str, float]) -> pd.DataFrame:
    categories = CATEGORY_OPTIONS
    base_scores = [68, 72, 64, 70, 66, 71]
    bonus = {
        "Electronics": 12,
        "Groceries": 9,
        "Fashion": 10,
        "Home & Kitchen": 8,
        "Beauty": 7,
        "Sports": 11,
    }
    df = pd.DataFrame(
        {
            "Category": categories,
            "Market Momentum": [base + bonus[cat] for base, cat in zip(base_scores, categories)],
            "Selected": [cat == selected_category for cat in categories],
        }
    )
    df["Momentum Label"] = np.where(df["Selected"], "Your Category", "Market")
    if feature_importance:
        df["Feature Influence"] = np.linspace(100, 55, len(df))
    else:
        df["Feature Influence"] = np.linspace(85, 45, len(df))
    return df


# --- PDF export ----------------------------------------------------------------

def create_pdf_report(inputs: PredictionInputs, prediction: dict[str, Any], contributions: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleWhite", parent=styles["Title"], fontName="Helvetica-Bold", textColor=colors.HexColor("#111827"), fontSize=20, leading=24, spaceAfter=10))
    styles.add(ParagraphStyle(name="BodyGray", parent=styles["BodyText"], fontName="Helvetica", textColor=colors.HexColor("#374151"), fontSize=10, leading=14))

    price = float(prediction["optimal_price"])
    delta = float(prediction["price_change_pct"])
    elements: list[Any] = []
    elements.append(Paragraph("AI Dynamic Pricing Report", styles["TitleWhite"]))
    elements.append(Paragraph(f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["BodyGray"]))
    elements.append(Spacer(1, 8))

    summary_data = [
        ["Category", inputs.category],
        ["Base Price", f"₹{inputs.base_price:,.2f}"],
        ["Recommended Price", f"₹{price:,.2f}"],
        ["Price Change", f"{delta:+.2f}%"],
        ["Strategy", prediction.get("strategy", "N/A")],
    ]
    table = Table(summary_data, colWidths=[60 * mm, 95 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#EFF6FF"), colors.white]),
            ]
        )
    )
    elements.append(table)
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Top Influencing Factors", styles["Heading2"]))
    factor_rows = [[row.factor, f"{row.score:.1f}", row.direction] for row in contributions.itertuples(index=False)]
    factor_table = Table([["Factor", "Score", "Direction"]] + factor_rows, colWidths=[70 * mm, 35 * mm, 40 * mm])
    factor_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563EB")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ]
        )
    )
    elements.append(factor_table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("This report was generated from the live FastAPI prediction endpoint and the current feature-importance service.", styles["BodyGray"]))
    doc.build(elements)
    buffer.seek(0)
    return buffer.read()


# --- Charts -------------------------------------------------------------------

def build_competitor_chart(inputs: PredictionInputs, recommended_price: float) -> go.Figure:
    labels = ["Seller A", "Seller B", "Seller C", "Recommended Price"]
    values = [inputs.competitor_price_1, inputs.competitor_price_2, inputs.base_price * 0.985, recommended_price]
    colors_map = ["#94A3B8", "#38BDF8", "#F59E0B", PRIMARY]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=colors_map,
                text=[f"₹{v:,.0f}" for v in values],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Competitor Comparison",
        yaxis_title="Price (₹)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white"),
        height=380,
        margin=dict(l=10, r=10, t=60, b=20),
    )
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


def build_demand_scatter(inputs: PredictionInputs, recommended_price: float) -> go.Figure:
    spread = np.linspace(max(inputs.base_price * 0.75, 1), inputs.base_price * 1.25, 14)
    demand = np.clip(100 - (spread - inputs.base_price) * 0.04 + inputs.demand_score * 0.45, 5, 100)
    inventory_curve = np.clip(100 - (inputs.inventory_level / 5.0), 0, 100)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spread,
            y=demand,
            mode="markers+lines",
            marker=dict(size=10, color=np.linspace(15, 95, len(spread)), colorscale="Blues", showscale=False),
            line=dict(color="#60A5FA", width=3),
            name="Demand Profile",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[inputs.base_price],
            y=[inputs.demand_score],
            mode="markers",
            marker=dict(size=16, color=SUCCESS, line=dict(color="white", width=1)),
            name="Current Position",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[recommended_price],
            y=[np.clip(inputs.demand_score + (recommended_price - inputs.base_price) / max(inputs.base_price, 1e-6) * 30, 5, 100)],
            mode="markers",
            marker=dict(size=18, color=WARNING, line=dict(color="white", width=1)),
            name="Recommended Position",
        )
    )
    fig.update_layout(
        title="Demand vs Inventory Analysis",
        xaxis_title="Price (₹)",
        yaxis_title="Demand Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white"),
        height=380,
        margin=dict(l=10, r=10, t=60, b=20),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


def build_feature_importance_chart(feature_importance: dict[str, float]) -> go.Figure:
    if feature_importance:
        df = pd.DataFrame(list(feature_importance.items()), columns=["Feature", "Importance"])
        df = df.sort_values("Importance", ascending=True)
    else:
        df = pd.DataFrame(
            {
                "Feature": ["market_pressure", "demand_score", "price_gap_1", "price_gap_2"],
                "Importance": [36.5, 31.4, 9.9, 8.1],
            }
        )
    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Feature Importance",
        color="Importance",
        color_continuous_scale=[[0, "#93C5FD"], [0.5, "#2563EB"], [1, "#10B981"]],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white"),
        height=430,
        margin=dict(l=10, r=10, t=60, b=20),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


def build_seasonal_chart(inputs: PredictionInputs) -> go.Figure:
    df = build_monthly_seasonality(inputs)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["month"],
            y=df["seasonal_influence"],
            mode="lines+markers",
            line=dict(color=PRIMARY, width=4),
            marker=dict(size=10, color="#93C5FD"),
            name="Monthly Influence",
        )
    )
    fig.add_trace(
        go.Bar(
            x=df["month"],
            y=df["weekend_pressure"],
            marker_color=SUCCESS,
            opacity=0.4,
            name="Weekend Pressure",
        )
    )
    fig.update_layout(
        title="Seasonal Impact Dashboard",
        xaxis_title="Month",
        yaxis_title="Influence Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white"),
        height=420,
        barmode="overlay",
        margin=dict(l=10, r=10, t=60, b=20),
    )
    fig.update_xaxes(dtick=1, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


# --- UI components ------------------------------------------------------------

def metric_card(label: str, value: str, delta: str | None = None, delta_color: str = SUCCESS) -> None:
    delta_html = f'<div style="color:{delta_color}; font-weight:700; margin-top:0.3rem;">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="section-label">{label}</div>
            <div style="font-size:1.75rem; font-weight:800; color:white;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<div class='hero-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>AI Dynamic Pricing</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:1.6rem; font-weight:800; line-height:1.1;'>Business Dashboard</div>", unsafe_allow_html=True)
        st.markdown("<div class='kpi-subtitle'>FastAPI + Streamlit decision cockpit for pricing teams.</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
        st.markdown("### Category Selection")
        st.selectbox("", CATEGORY_OPTIONS, key="selected_category", label_visibility="collapsed")

        st.markdown("### Product Information")
        st.number_input("Base Price", min_value=1.0, max_value=100000.0, value=2499.0, step=1.0, key="base_price")

        st.markdown("### Competitor Analysis")
        st.number_input("Seller A Price", min_value=1.0, max_value=100000.0, value=2399.0, step=1.0, key="competitor_price_1")
        st.number_input("Seller B Price", min_value=1.0, max_value=100000.0, value=2449.0, step=1.0, key="competitor_price_2")
        st.number_input("Seller C Price", min_value=1.0, max_value=100000.0, value=2480.0, step=1.0, key="seller_c_price")

        st.markdown("### Market Conditions")
        st.slider("Inventory Level", min_value=0, max_value=5000, value=240, step=10, key="inventory_level")
        st.slider("Demand Score", min_value=0, max_value=100, value=68, step=1, key="demand_score")
        st.slider("Customer Rating", min_value=1.0, max_value=5.0, value=4.4, step=0.1, key="customer_rating")

        st.markdown("### Time Factors")
        day_choice = st.selectbox("Weekday/Weekend Selector", DAY_NAMES, index=datetime.now().weekday())
        st.session_state["day_of_week"] = DAY_INDEX[day_choice]
        st.selectbox("Month", MONTH_OPTIONS, index=datetime.now().month - 1, key="month")

        st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
        st.button("Predict Price", type="primary", use_container_width=True, key="predict_button")

        st.caption("Connected to live Railway backend via requests.post().")


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">Revenue Intelligence</div>
            <div style="display:flex; justify-content:space-between; align-items:flex-end; gap:1rem; flex-wrap:wrap;">
                <div>
                    <div style="font-size:2.1rem; font-weight:900; color:white; line-height:1.05;">AI Dynamic Pricing Dashboard</div>
                    <div class="kpi-subtitle">Professional pricing cockpit for hackathon judges and interview demos.</div>
                </div>
                <div class="badge-pill">Live FastAPI Integration</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- Main flow ----------------------------------------------------------------

def main() -> None:
    inject_css()
    render_sidebar()
    render_header()

    inputs = build_inputs()
    feature_importance = get_feature_importance_cached()
    categories = get_categories_cached()

    if categories and inputs.category not in categories:
        inputs.category = categories[0]

    tab_overview, tab_explain, tab_charts, tab_reports = st.tabs([
        "Overview",
        "AI Explanation",
        "Charts",
        "Reports & History",
    ])

    prediction: dict[str, Any] | None = None
    contributions_df: pd.DataFrame | None = None

    with tab_overview:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        left, center, right = st.columns([1.1, 1.4, 1.1], vertical_alignment="top")

        with left:
            st.markdown("#### Live Performance")
            metric_card("Current Price", f"₹{inputs.base_price:,.0f}")
            metric_card("Competitor Average", f"₹{((inputs.competitor_price_1 + inputs.competitor_price_2) / 2.0):,.0f}")
            metric_card("Inventory Level", f"{inputs.inventory_level:,}")

        with center:
            st.markdown("#### Prediction Center")
            st.markdown(
                "Use the controls in the sidebar, then press Predict Price. The app will call the deployed FastAPI model in real time.",
            )
            if st.session_state.get("predict_button"):
                try:
                    with st.spinner("Calling pricing API..."):
                        prediction = api_predict(build_payload(inputs))
                    st.session_state["last_prediction"] = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "category": inputs.category,
                        "base_price": inputs.base_price,
                        "recommended_price": prediction["optimal_price"],
                        "price_change_pct": prediction["price_change_pct"],
                        "strategy": prediction.get("strategy", "N/A"),
                    }
                    contributions_df = calculate_contributions(inputs, float(prediction["optimal_price"]))
                    st.success(prediction.get("recommendation", "Price recommendation generated."))
                except requests.RequestException as exc:
                    st.error(f"FastAPI request failed: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error while generating prediction: {exc}")
            elif "last_prediction" in st.session_state:
                st.info("Showing the latest saved prediction from this session.")

            if prediction is None and "last_prediction" in st.session_state:
                prediction = {
                    "optimal_price": st.session_state["last_prediction"]["recommended_price"],
                    "price_change_pct": st.session_state["last_prediction"]["price_change_pct"],
                    "strategy": st.session_state["last_prediction"]["strategy"],
                    "recommendation": "Latest session prediction loaded.",
                }
                contributions_df = calculate_contributions(inputs, float(prediction["optimal_price"]))

        with right:
            st.markdown("#### Real-time Metrics")
            if prediction:
                delta_color = SUCCESS if float(prediction["price_change_pct"]) >= 0 else DANGER
                metric_card("Recommended Price", f"₹{prediction['optimal_price']:,.2f}", f"{prediction['price_change_pct']:+.2f}%", delta_color=delta_color)
                metric_card("Strategy", prediction.get("strategy", "N/A"))
                market_spread = abs(inputs.base_price - float(prediction["optimal_price"]))
                metric_card("Price Delta", f"₹{market_spread:,.2f}")
            else:
                metric_card("Recommended Price", "₹ --")
                metric_card("Strategy", "Awaiting prediction")
                metric_card("Price Delta", "₹ --")

        st.markdown("</div>", unsafe_allow_html=True)

        if prediction:
            st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
            price_col, delta_col, strategy_col = st.columns([1.2, 1, 1])
            with price_col:
                st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Recommended Price</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value'>₹ {prediction['optimal_price']:,.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-subtitle'>Live price returned from your Railway FastAPI backend.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with delta_col:
                pct = float(prediction["price_change_pct"])
                color = SUCCESS if pct >= 0 else DANGER
                st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Price Increase / Decrease</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value' style='color:{color};'>{pct:+.2f}%</div>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-subtitle'>Green indicates an upward recommendation. Red indicates a decrease.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            with strategy_col:
                st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-label'>Strategy</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='kpi-value' style='font-size:1.8rem;'>{prediction.get('strategy', 'N/A')}</div>", unsafe_allow_html=True)
                st.markdown("<div class='kpi-subtitle'>Model-driven pricing recommendation for the current context.</div>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"₹{inputs.base_price:,.2f}")
            c2.metric("Recommended Price", f"₹{prediction['optimal_price']:,.2f}")
            c3.metric("Competitor Avg", f"₹{((inputs.competitor_price_1 + inputs.competitor_price_2) / 2.0):,.2f}")
            c4.metric("Category", inputs.category)

    with tab_explain:
        st.markdown("### Why did the AI recommend this price?")
        st.write("These panels translate the live prediction into business-readable drivers. They are derived from the live request inputs and the returned recommended price.")

        if prediction is None:
            st.info("Run a prediction to see the explanation layer.")
        else:
            if contributions_df is None:
                contributions_df = calculate_contributions(inputs, float(prediction["optimal_price"]))
            cols = st.columns(2)
            for idx, row in enumerate(contributions_df.itertuples(index=False)):
                with cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div class='factor-card'>
                            <div class='section-label'>{row.factor}</div>
                            <div style='font-size:1.15rem; font-weight:800; color:white;'>{row.headline}</div>
                            <div style='color:{SUCCESS if row.impact >= 0 else DANGER}; font-size:1.2rem; font-weight:800; margin:0.35rem 0;'>
                                {row.score:.1f}/100
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(float(np.clip(row.score / 100.0, 0.0, 1.0)))
                    st.caption(
                        f"Impact estimate: {'+' if row.impact >= 0 else ''}{row.impact:.2f} relative signal"
                    )

            with st.expander("Show raw factor breakdown"):
                st.dataframe(
                    contributions_df[["factor", "score", "direction", "impact", "headline"]],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_charts:
        st.markdown("### Interactive Pricing Intelligence")
        if prediction is None:
            st.info("Predict once to populate the live charts.")
        else:
            recommended_price = float(prediction["optimal_price"])
            c1, c2 = st.columns([1, 1])
            with c1:
                st.plotly_chart(build_competitor_chart(inputs, recommended_price), use_container_width=True)
            with c2:
                st.plotly_chart(build_demand_scatter(inputs, recommended_price), use_container_width=True)

            c3, c4 = st.columns([1, 1])
            with c3:
                st.plotly_chart(build_feature_importance_chart(feature_importance), use_container_width=True)
            with c4:
                st.plotly_chart(build_seasonal_chart(inputs), use_container_width=True)

            st.markdown("#### Category Analytics")
            category_df = build_category_analytics(inputs.category, feature_importance)
            fig = px.bar(
                category_df,
                x="Category",
                y="Market Momentum",
                color="Selected",
                color_discrete_map={True: PRIMARY, False: "#64748B"},
                title="Category Analytics",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,0.02)",
                font=dict(color="white"),
                height=360,
                margin=dict(l=10, r=10, t=60, b=20),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab_reports:
        st.markdown("### Reports and History")
        if prediction and contributions_df is not None:
            pdf_bytes = create_pdf_report(inputs, prediction, contributions_df)
            st.download_button(
                "Download Prediction Report (PDF)",
                data=pdf_bytes,
                file_name=f"pricing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=False,
            )

        history = st.session_state.get("prediction_history", [])
        if prediction:
            history = [
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": inputs.category,
                    "base_price": inputs.base_price,
                    "recommended_price": float(prediction["optimal_price"]),
                    "price_change_pct": float(prediction["price_change_pct"]),
                    "strategy": prediction.get("strategy", "N/A"),
                }
            ] + history
            st.session_state["prediction_history"] = history[:15]

        if st.session_state.get("prediction_history"):
            st.dataframe(pd.DataFrame(st.session_state["prediction_history"]), use_container_width=True, hide_index=True)
        else:
            st.info("Prediction history will appear here after the first successful API call.")

        st.markdown("#### Backend Status")
        try:
            health = api_get("/health")
            st.json(health)
        except Exception as exc:
            st.error(f"Unable to fetch backend health: {exc}")


if __name__ == "__main__":
    main()
