from __future__ import annotations

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
    page_title="Retail AI Dynamic Pricing Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = os.getenv("API_URL", "https://dynamic-pricing-ai-production.up.railway.app").rstrip("/")

SIDEBAR_CATEGORIES = [
    "📱 Electronics",
    "🛒 Groceries",
    "👕 Fashion",
    "🏠 Home & Kitchen",
    "⚽ Sports",
    "💄 Beauty",
]
BACKEND_CATEGORY_MAP = {
    "📱 Electronics": "Electronics",
    "🛒 Groceries": "Groceries",
    "👕 Fashion": "Clothing",
    "🏠 Home & Kitchen": "Home & Garden",
    "⚽ Sports": "Sports",
    "💄 Beauty": "Clothing",
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_INDEX = {name: idx for idx, name in enumerate(DAY_NAMES)}
MONTHS = list(range(1, 13))

PRIMARY = "#2563EB"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
PANEL = "rgba(17, 24, 39, 0.72)"
BORDER = "rgba(148, 163, 184, 0.18)"
TEXT = "#E5E7EB"
MUTED = "#94A3B8"


@dataclass
class CategoryDefaults:
    base_price: float
    seller_a: float
    seller_b: float
    seller_c: float
    inventory_level: int
    demand_score: int
    customer_rating: float


DEFAULTS = {
    "📱 Electronics": CategoryDefaults(2499, 2399, 2449, 2480, 220, 68, 4.4),
    "🛒 Groceries": CategoryDefaults(249, 239, 244, 242, 1200, 55, 4.2),
    "👕 Fashion": CategoryDefaults(1799, 1699, 1749, 1720, 320, 61, 4.5),
    "🏠 Home & Kitchen": CategoryDefaults(1499, 1449, 1489, 1469, 410, 57, 4.3),
    "⚽ Sports": CategoryDefaults(2199, 2099, 2149, 2119, 180, 63, 4.4),
    "💄 Beauty": CategoryDefaults(999, 949, 979, 969, 500, 58, 4.6),
}


def _safe_get(mapping: Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, dict):
        return mapping.get(key, default)
    return default


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background:
                radial-gradient(circle at 8% 10%, rgba(37,99,235,0.24), transparent 25%),
                radial-gradient(circle at 88% 2%, rgba(16,185,129,0.16), transparent 22%),
                radial-gradient(circle at 70% 92%, rgba(245,158,11,0.10), transparent 24%),
                linear-gradient(180deg, #050816 0%, #0b1120 45%, #0f172a 100%);
            color: {TEXT};
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(8,12,24,0.96), rgba(10,16,30,0.98));
            border-right: 1px solid {BORDER};
        }}

        .glass-card, .metric-card, .insight-card, .report-card, .info-card {{
            background: {PANEL};
            border: 1px solid {BORDER};
            border-radius: 22px;
            box-shadow: 0 18px 50px rgba(0,0,0,0.22);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
        }}

        .hero {{ padding: 1.25rem 1.35rem; background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(16,185,129,0.10)); }}
        .section-label {{ text-transform: uppercase; letter-spacing: 0.16em; color: {MUTED}; font-size: 0.72rem; font-weight: 800; margin-bottom: 0.35rem; }}
        .hero-title {{ font-size: 2.15rem; line-height: 1.02; font-weight: 900; letter-spacing: -0.05em; color: white; }}
        .hero-subtitle {{ color: {MUTED}; font-size: 0.98rem; margin-top: 0.45rem; }}
        .kpi-value {{ font-size: 2.15rem; line-height: 1.0; font-weight: 900; letter-spacing: -0.05em; color: white; }}
        .kpi-delta {{ font-size: 0.96rem; font-weight: 800; margin-top: 0.35rem; }}
        .metric-card, .insight-card, .report-card, .info-card {{ padding: 1rem 1.05rem; }}
        .small-label {{ color: {MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 800; margin-bottom: 0.3rem; }}

        .stButton > button {{
            background: linear-gradient(135deg, #2563EB, #1D4ED8);
            color: white;
            border: 0;
            border-radius: 16px;
            font-weight: 900;
            padding: 0.92rem 1.2rem;
            box-shadow: 0 18px 28px rgba(37,99,235,0.25);
        }}

        .stButton > button:hover {{ color: white; transform: translateY(-1px); }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 0.5rem; }}
        .stTabs [data-baseweb="tab"] {{ background: rgba(15,23,42,0.82); border: 1px solid {BORDER}; color: #cbd5e1; border-radius: 999px; padding: 0.7rem 1rem; font-weight: 800; }}
        .stTabs [aria-selected="true"] {{ background: linear-gradient(135deg, rgba(37,99,235,0.28), rgba(16,185,129,0.18)); color: white; }}
        .stMetric {{ background: rgba(15,23,42,0.36); border: 1px solid rgba(148,163,184,0.15); border-radius: 18px; padding: 0.6rem 0.65rem; }}
        [data-testid="stMetricValue"] {{ color: white; font-weight: 900; }}
        [data-testid="stMetricDelta"] {{ font-weight: 800; }}
        .factor-badge {{ display: inline-block; padding: 0.3rem 0.7rem; border-radius: 999px; background: rgba(37,99,235,0.16); border: 1px solid rgba(37,99,235,0.25); color: #dbeafe; font-size: 0.78rem; font-weight: 800; }}
        .soft-divider {{ height: 1px; background: linear-gradient(90deg, transparent, rgba(148,163,184,0.4), transparent); margin: 1rem 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# API helpers
# -----------------------------------------------------------------------------

def api_get(path: str, timeout: int = 15) -> Any:
    response = requests.get(f"{API_URL}{path}", timeout=timeout)
    response.raise_for_status()
    return response.json()


def api_predict(payload: dict[str, Any], timeout: int = 20) -> dict[str, Any]:
    response = requests.post(f"{API_URL}/predict-price", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=300, show_spinner=False)
def get_feature_importance() -> dict[str, float]:
    try:
        data = api_get("/feature-importance")
        return {str(k): float(v) for k, v in data.items()}
    except Exception:
        return {}


@st.cache_data(ttl=300, show_spinner=False)
def get_categories() -> list[str]:
    try:
        payload = api_get("/categories")
        categories = payload.get("categories", [])
        return [str(item) for item in categories] if categories else []
    except Exception:
        return []


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def backend_category(category: str) -> str:
    return BACKEND_CATEGORY_MAP.get(category, "Electronics")


def active_defaults(category: str) -> CategoryDefaults:
    return DEFAULTS[category]


def set_category_defaults(category: str) -> None:
    defaults = active_defaults(category)
    st.session_state["base_price"] = float(defaults.base_price)
    st.session_state["seller_a"] = float(defaults.seller_a)
    st.session_state["seller_b"] = float(defaults.seller_b)
    st.session_state["seller_c"] = float(defaults.seller_c)
    st.session_state["inventory_level"] = int(defaults.inventory_level)
    st.session_state["demand_score"] = int(defaults.demand_score)
    st.session_state["customer_rating"] = float(defaults.customer_rating)
    st.session_state["month"] = datetime.now().month
    st.session_state["day_of_week"] = datetime.now().weekday()
    st.session_state["holiday_season"] = False


def build_payload(category: str, inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "category": backend_category(category),
        "base_price": float(inputs["base_price"]),
        "competitor_price_1": float(inputs["seller_a"]),
        "competitor_price_2": float(inputs["seller_b"]),
        "inventory_level": int(inputs["inventory_level"]),
        "demand_score": float(inputs["demand_score"]) / 100.0,
        "day_of_week": int(inputs["day_of_week"]),
        "month": int(inputs["month"]),
        "customer_rating": float(inputs["customer_rating"]),
        "is_weekend": int(inputs["day_of_week"]) >= 5,
        "is_holiday_season": bool(inputs["holiday_season"]),
    }


def current_inputs(category: str) -> dict[str, Any]:
    return {
        "category": category,
        "base_price": float(st.session_state["base_price"]),
        "seller_a": float(st.session_state["seller_a"]),
        "seller_b": float(st.session_state["seller_b"]),
        "seller_c": float(st.session_state["seller_c"]),
        "inventory_level": int(st.session_state["inventory_level"]),
        "demand_score": int(st.session_state["demand_score"]),
        "customer_rating": float(st.session_state["customer_rating"]),
        "month": int(st.session_state["month"]),
        "day_of_week": int(st.session_state["day_of_week"]),
        "holiday_season": bool(st.session_state.get("holiday_season", False)),
    }


def compute_insight_data(category: str, inputs: dict[str, Any], prediction: dict[str, Any]) -> pd.DataFrame:
    base_price = float(_safe_get(inputs, "base_price", 0) or 0)
    pred_price = float(_safe_get(prediction, "optimal_price", 0) or 0)
    comp_avg = (float(_safe_get(inputs, "seller_a", 0) or 0) + float(_safe_get(inputs, "seller_b", 0) or 0) + float(_safe_get(inputs, "seller_c", 0) or 0)) / 3.0
    demand_score = float(_safe_get(inputs, "demand_score", 0) or 0)
    inventory = float(_safe_get(inputs, "inventory_level", 0) or 0)
    customer_rating = float(_safe_get(inputs, "customer_rating", 0) or 0)
    month = int(_safe_get(inputs, "month", datetime.now().month) or datetime.now().month)
    holiday = bool(_safe_get(inputs, "holiday_season", False))

    competitor_pressure = np.clip((comp_avg - base_price) / max(base_price, 1e-6), -0.8, 0.8)
    inventory_risk = np.clip((250 - inventory) / 250.0, -1.0, 1.0)
    demand_trend = np.clip((demand_score - 50) / 50.0, -1.0, 1.0)
    seasonality = 0.45 if holiday else (0.22 if month in [6, 7, 8] else 0.05)
    market_position = "Premium" if pred_price > comp_avg else "Competitive"

    rows = [
        ("Pricing Strategy", str(prediction.get("strategy", "N/A")), SUCCESS if pred_price >= base_price else DANGER),
        ("Market Position", market_position, PRIMARY),
        ("Competitor Pressure", f"{competitor_pressure:+.2f}", WARNING if competitor_pressure > 0 else SUCCESS),
        ("Inventory Risk", f"{inventory_risk:+.2f}", WARNING if inventory_risk > 0.3 else SUCCESS),
        ("Demand Trend", f"{demand_trend:+.2f}", SUCCESS if demand_trend > 0 else DANGER),
        ("Business Recommendation", str(_safe_get(prediction, "recommendation", "Live recommendation generated.")), PRIMARY),
    ]
    df = pd.DataFrame(rows, columns=["title", "value", "color"])
    df["demand_score"] = demand_score
    df["customer_rating"] = customer_rating
    df["seasonality"] = seasonality
    df["category"] = category
    return df


def compute_business_analytics(category: str, inputs: dict[str, Any], prediction: dict[str, Any]) -> pd.DataFrame:
    base_price = float(_safe_get(inputs, "base_price", 0) or 0)
    recommended = float(_safe_get(prediction, "optimal_price", 0) or 0)
    comp_avg = (float(_safe_get(inputs, "seller_a", 0) or 0) + float(_safe_get(inputs, "seller_b", 0) or 0) + float(_safe_get(inputs, "seller_c", 0) or 0)) / 3.0
    inventory = float(_safe_get(inputs, "inventory_level", 0) or 0)
    demand = float(_safe_get(inputs, "demand_score", 0) or 0)
    margin = ((recommended - (base_price * 0.72)) / max(recommended, 1e-6)) * 100.0
    revenue_uplift = ((recommended - base_price) * max(demand, 0.1) * 10.0)
    competitive_gap = ((recommended - comp_avg) / max(comp_avg, 1e-6)) * 100.0
    inventory_health = np.clip(100.0 - (inventory / 25.0), 0.0, 100.0)
    df = pd.DataFrame(
        [
            ("Revenue Impact", f"₹{revenue_uplift:,.0f}", SUCCESS if revenue_uplift >= 0 else DANGER),
            ("Expected Margin", f"{margin:.1f}%", SUCCESS if margin > 20 else WARNING),
            ("Competitive Gap", f"{competitive_gap:+.1f}%", SUCCESS if competitive_gap <= 0 else WARNING),
            ("Inventory Health", f"{inventory_health:.0f}/100", SUCCESS if inventory_health > 50 else DANGER),
        ],
        columns=["metric", "value", "color"],
    )
    df["category"] = category
    return df


# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------

def build_feature_importance_chart(feature_importance: dict[str, float]) -> go.Figure:
    if not feature_importance:
        feature_importance = {
            "demand_score": 31.4,
            "market_pressure": 24.9,
            "inventory_level": 14.2,
            "competitor_price_1": 10.8,
            "competitor_price_2": 8.9,
            "customer_rating": 5.7,
            "seasonality": 4.1,
        }
    df = pd.DataFrame(list(feature_importance.items()), columns=["feature", "importance"])
    df = df.sort_values("importance", ascending=True)
    fig = px.bar(
        df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance",
        color="importance",
        color_continuous_scale=[[0, "#93C5FD"], [0.5, "#2563EB"], [1, "#10B981"]],
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)", font=dict(color="white"), height=470, margin=dict(l=10, r=10, t=60, b=20), coloraxis_showscale=False)
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


def build_competitor_chart(inputs: dict[str, Any], prediction: dict[str, Any]) -> go.Figure:
    base_price = float(_safe_get(inputs, "base_price", 0) or 0)
    seller_a = float(_safe_get(inputs, "seller_a", 0) or 0)
    seller_b = float(_safe_get(inputs, "seller_b", 0) or 0)
    seller_c = float(_safe_get(inputs, "seller_c", 0) or 0)
    recommended = float(_safe_get(prediction, "optimal_price", 0) or 0)
    df = pd.DataFrame(
        {
            "source": ["Seller A", "Seller B", "Seller C", "Recommended Price"],
            "price": [seller_a, seller_b, seller_c, recommended],
            "highlight": [False, False, False, True],
        }
    )
    fig = px.bar(df, x="source", y="price", color="highlight", color_discrete_map={False: "#64748B", True: PRIMARY}, title="Competitor Comparison Chart", text=df["price"].map(lambda v: f"₹{v:,.0f}"))
    fig.update_traces(textposition="outside")
    fig.add_hline(y=base_price, line_dash="dot", line_color="#F59E0B", annotation_text="Base Price")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)", font=dict(color="white"), height=430, showlegend=False, margin=dict(l=10, r=10, t=60, b=20))
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.15)")
    return fig


def build_demand_inventory_chart(inputs: dict[str, Any]) -> go.Figure:
    inventory = float(_safe_get(inputs, "inventory_level", 0) or 0)
    demand = float(_safe_get(inputs, "demand_score", 0) or 0)
    df = pd.DataFrame(
        [
            ("Overstock", 0.18, 0.32),
            ("Balanced", 0.45, 0.55),
            ("High Demand", 0.72, 0.80),
            ("Stock Risk", 0.85, 0.25),
        ],
        columns=["zone", "inventory", "demand"],
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["inventory"], y=df["demand"], mode="markers+text", text=df["zone"], textposition="top center", marker=dict(size=18, color=["#94A3B8", "#38BDF8", SUCCESS, DANGER], line=dict(color="white", width=1.2)), name="Business Zones"))
    fig.add_trace(go.Scatter(x=[inventory / 500.0], y=[demand / 100.0], mode="markers", marker=dict(size=22, color=WARNING, line=dict(color="white", width=1.5)), name="Current Position"))
    fig.update_layout(title="Demand vs Inventory Analysis", xaxis_title="Inventory (normalized)", yaxis_title="Demand (normalized)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)", font=dict(color="white"), height=430, margin=dict(l=10, r=10, t=60, b=20))
    fig.add_vline(x=0.5, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig.add_hline(y=0.5, line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def build_monthly_trend_chart(inputs: dict[str, Any], prediction: dict[str, Any]) -> go.Figure:
    base_price = float(_safe_get(inputs, "base_price", 0) or 0)
    recommended = float(_safe_get(prediction, "optimal_price", 0) or 0)
    months = list(range(1, 13))
    base_series = [base_price for _ in months]
    seasonal_factors = [0.96, 0.94, 0.98, 1.00, 1.03, 1.07, 1.08, 1.09, 1.06, 1.11, 1.18, 1.22]
    recommended_series = [recommended * factor for factor in seasonal_factors]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=base_series, mode="lines", line=dict(color="#94A3B8", width=3, dash="dot"), name="Base Price"))
    fig.add_trace(go.Scatter(x=months, y=recommended_series, mode="lines+markers", line=dict(color=PRIMARY, width=4), marker=dict(size=8, color="#93C5FD"), name="Recommended Price"))
    fig.update_layout(title="Monthly Price Trend", xaxis_title="Month", yaxis_title="Price (₹)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(255,255,255,0.02)", font=dict(color="white"), height=430, margin=dict(l=10, r=10, t=60, b=20))
    fig.update_xaxes(dtick=1, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


# -----------------------------------------------------------------------------
# PDF and UI components
# -----------------------------------------------------------------------------

def build_pdf_report(category: str, inputs: dict[str, Any], prediction: dict[str, Any], factors: pd.DataFrame) -> bytes:
    try:
        if isinstance(factors, pd.DataFrame):
            print(factors.columns.tolist())
            st.write("Factors columns:", factors.columns.tolist())
        else:
            print([])
            st.write("Factors columns:", [])

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=15 * mm,
            leftMargin=15 * mm,
            topMargin=15 * mm,
            bottomMargin=15 * mm,
        )
        styles = getSampleStyleSheet()
        styles.add(
            ParagraphStyle(
                name="ReportTitle",
                parent=styles["Title"],
                fontName="Helvetica-Bold",
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#111827"),
            )
        )
        styles.add(
            ParagraphStyle(
                name="ReportBody",
                parent=styles["BodyText"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#374151"),
            )
        )

        rows = [
            ["Category", category],
            ["Base Price", f"₹{float(_safe_get(inputs, 'base_price', 0) or 0):,.2f}"],
            ["Recommended Price", f"₹{float(_safe_get(prediction, 'optimal_price', 0) or 0):,.2f}"],
            ["Price Change %", f"{float(_safe_get(prediction, 'price_change_pct', 0) or 0):+.2f}%"],
            ["Strategy", str(_safe_get(prediction, "strategy", "N/A"))],
            ["Recommendation", str(_safe_get(prediction, "recommendation", "N/A"))],
        ]
        summary = Table(rows, colWidths=[55 * mm, 110 * mm])
        summary.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DBEAFE")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#EFF6FF"), colors.white]),
                ]
            )
        )

        if not isinstance(factors, pd.DataFrame) or factors.empty:
            factor_rows = [["No factor analysis available"]]
            factor_table = Table([["Factor Analysis"]] + factor_rows, colWidths=[150 * mm])
        else:
            available_columns = [str(column) for column in factors.columns.tolist()]
            if {"Factor", "Score", "Direction"}.issubset(factors.columns):
                factor_rows = [
                    [
                        row.get("Factor", "Unknown"),
                        f"{float(row.get('Score', 0) or 0):.1f}",
                        row.get("Direction", "N/A"),
                    ]
                    for _, row in factors.iterrows()
                ]
                factor_table = Table([["Factor", "Score", "Direction"]] + factor_rows, colWidths=[70 * mm, 30 * mm, 40 * mm])
            else:
                factor_rows = factors.fillna("").astype(str).values.tolist()
                if not factor_rows:
                    factor_rows = [["No factor analysis available"]]
                    available_columns = ["Factor Analysis"]
                factor_table = Table([available_columns] + factor_rows, colWidths=None)

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

        elements: list[Any] = [
            Paragraph("Retail AI Dynamic Pricing Report", styles["ReportTitle"]),
            Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["ReportBody"]),
            Spacer(1, 6),
            summary,
            Spacer(1, 10),
            Paragraph("Top Influencing Factors", styles["Heading2"]),
            factor_table,
            Spacer(1, 8),
            Paragraph(
                "This report was generated from live API-backed inputs and the current backend prediction response.",
                styles["ReportBody"],
            ),
        ]
        doc.build(elements)
        buffer.seek(0)
        return buffer.read()
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return b""


def metric_card(title: str, value: str, delta: str | None = None, delta_color: str = SUCCESS, icon: str = "") -> None:
    delta_html = f"<div class='kpi-delta' style='color:{delta_color};'>{delta}</div>" if delta else ""
    st.markdown(f"""<div class="metric-card glass-card"><div class="small-label">{icon} {title}</div><div class="kpi-value">{value}</div>{delta_html}</div>""", unsafe_allow_html=True)


def insight_card(title: str, value: str, color: str) -> None:
    st.markdown(f"""<div class="insight-card glass-card"><div class="small-label">{title}</div><div style="font-size:1.08rem; font-weight:800; color:{color}; line-height:1.35;">{value}</div></div>""", unsafe_allow_html=True)


def initialize_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state["prediction_history"] = []
    if "_active_category" not in st.session_state:
        st.session_state["_active_category"] = SIDEBAR_CATEGORIES[0]
    if "base_price" not in st.session_state:
        set_category_defaults(st.session_state["_active_category"])
    if "holiday_season" not in st.session_state:
        st.session_state["holiday_season"] = False


def render_sidebar() -> tuple[str, bool, dict[str, Any]]:
    with st.sidebar:
        st.markdown("""<div class="hero glass-card"><div class="section-label">Retail AI Platform</div><div class="hero-title">Dynamic Pricing Console</div><div class="hero-subtitle">Startup-style pricing cockpit for hackathons, interviews, and demos.</div></div>""", unsafe_allow_html=True)
        st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

        selected_category = st.radio("Category Navigation", SIDEBAR_CATEGORIES, index=SIDEBAR_CATEGORIES.index(st.session_state["_active_category"]), key="category_nav")
        if selected_category != st.session_state["_active_category"]:
            st.session_state["_active_category"] = selected_category
            set_category_defaults(selected_category)

        defaults = active_defaults(selected_category)
        st.markdown("### Pricing Inputs")
        base_price = st.number_input("Base Price", min_value=1.0, max_value=100000.0, value=float(defaults.base_price), step=1.0, key="base_price")
        seller_a = st.number_input("Seller A Price", min_value=1.0, max_value=100000.0, value=float(defaults.seller_a), step=1.0, key="seller_a")
        seller_b = st.number_input("Seller B Price", min_value=1.0, max_value=100000.0, value=float(defaults.seller_b), step=1.0, key="seller_b")
        seller_c = st.number_input("Seller C Price", min_value=1.0, max_value=100000.0, value=float(defaults.seller_c), step=1.0, key="seller_c")

        st.markdown("### Market Signals")
        inventory_level = st.slider("Inventory Level Slider", min_value=0, max_value=5000, value=int(defaults.inventory_level), step=10, key="inventory_level")
        demand_score = st.slider("Demand Score Slider", min_value=0, max_value=100, value=int(defaults.demand_score), step=1, key="demand_score")
        customer_rating = st.slider("Customer Rating Slider", min_value=1.0, max_value=5.0, value=float(defaults.customer_rating), step=0.1, key="customer_rating")

        st.markdown("### Time Factors")
        month = st.selectbox("Month Selector", MONTHS, index=datetime.now().month - 1, key="month")
        day_name = st.selectbox("Weekday / Weekend Selector", DAY_NAMES, index=datetime.now().weekday(), key="day_name")
        day_of_week = DAY_INDEX[day_name]
        holiday_season = st.checkbox("Holiday Season Toggle", key="holiday_season")

        st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("Predict Price", use_container_width=True)
        st.caption("Live requests are sent directly to the Railway backend.")

    inputs = {
        "category": selected_category,
        "base_price": base_price,
        "seller_a": seller_a,
        "seller_b": seller_b,
        "seller_c": seller_c,
        "inventory_level": inventory_level,
        "demand_score": demand_score,
        "customer_rating": customer_rating,
        "month": month,
        "day_of_week": day_of_week,
        "holiday_season": holiday_season,
    }
    return selected_category, predict_clicked, inputs


def render_top_metrics(inputs: dict[str, Any], prediction: dict[str, Any]) -> None:
    base_price = float(inputs["base_price"])
    comp_avg = (float(inputs["seller_a"]) + float(inputs["seller_b"]) + float(inputs["seller_c"])) / 3.0
    rec_price = float(_safe_get(prediction, "optimal_price", base_price) or base_price)
    demand_score = float(inputs["demand_score"])
    inventory_level = float(inputs["inventory_level"])
    competitive_position = "Premium" if rec_price > comp_avg else "Competitive"
    profit_opportunity = ((rec_price - base_price) / max(base_price, 1e-6)) * 100.0
    inventory_status = "Healthy" if inventory_level >= 150 else "Tight"

    cols = st.columns(6)
    with cols[0]:
        metric_card("Recommended Price", f"₹{rec_price:,.2f}", icon="💰")
    with cols[1]:
        metric_card("Current Price", f"₹{base_price:,.2f}", icon="🏷️")
    with cols[2]:
        metric_card("Profit Opportunity", f"{profit_opportunity:+.1f}%", delta="vs current price", delta_color=SUCCESS if profit_opportunity >= 0 else DANGER, icon="📈")
    with cols[3]:
        metric_card("Demand Score", f"{demand_score:.0f}/100", icon="🔥")
    with cols[4]:
        metric_card("Inventory Status", inventory_status, delta=f"{inventory_level:,.0f} units", delta_color=WARNING, icon="📦")
    with cols[5]:
        metric_card("Competitive Position", competitive_position, delta=f"Avg ₹{comp_avg:,.0f}", delta_color=PRIMARY, icon="⚔️")


def render_insights_section(inputs: dict[str, Any], prediction: dict[str, Any]) -> pd.DataFrame:
    insights = compute_insight_data(str(inputs["category"]), inputs, prediction)
    st.markdown("### AI Insights")
    cols = st.columns(3)
    for idx, row in enumerate(insights.itertuples(index=False)):
        with cols[idx % 3]:
            insight_card(str(row.title), str(row.value), str(row.color))
    return insights


def render_business_analytics(inputs: dict[str, Any], prediction: dict[str, Any]) -> None:
    analytics = compute_business_analytics(str(inputs["category"]), inputs, prediction)
    st.markdown("### Business Analytics")
    cols = st.columns(4)
    for idx, row in enumerate(analytics.itertuples(index=False)):
        with cols[idx]:
            st.markdown(f"""<div class="report-card glass-card"><div class="small-label">{row.metric}</div><div class="kpi-value" style="font-size:1.75rem; color:{row.color};">{row.value}</div></div>""", unsafe_allow_html=True)


def render_prediction_history() -> None:
    st.markdown("### Prediction History")
    history = st.session_state.get("prediction_history", [])
    if history:
        df = pd.DataFrame(history)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", data=df.to_csv(index=False).encode("utf-8"), file_name="prediction_history.csv", mime="text/csv")
    else:
        st.info("Prediction history will appear after your first successful API call.")


def render_report_section(category: str, inputs: dict[str, Any], prediction: dict[str, Any], insights: pd.DataFrame) -> None:
    st.markdown("### Reports")
    st.download_button("Download PDF Pricing Report", data=build_pdf_report(category, inputs, prediction, insights), file_name=f"pricing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")


# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------

def main() -> None:
    inject_css()
    initialize_state()

    selected_category, predict_clicked, inputs = render_sidebar()
    st.markdown("""<div class="hero glass-card"><div class="section-label">Revenue Intelligence</div><div class="hero-title">Premium AI Dynamic Pricing Dashboard</div><div class="hero-subtitle">Glassmorphism interface, live Railway integration, dynamic charts, and executive-ready business outputs.</div></div>""", unsafe_allow_html=True)

    categories = get_categories()
    if categories and backend_category(selected_category) not in categories:
        st.warning("Selected category maps to a backend label that is not available in the model categories returned by the API.")

    prediction: dict[str, Any] | None = None
    if predict_clicked:
        try:
            with st.spinner("Requesting live prediction from Railway..."):
                payload = build_payload(selected_category, inputs)
                st.write(payload)
                prediction = api_predict(payload)
            st.success("Prediction generated successfully from the Railway API.")
            st.session_state["prediction_history"].insert(
                0,
                {
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Category": selected_category,
                    "Base Price": inputs["base_price"],
                    "Predicted Price": float(_safe_get(prediction, "optimal_price", inputs["base_price"]) or inputs["base_price"]),
                    "Strategy": prediction.get("strategy", "N/A"),
                },
            )
            st.session_state["prediction_history"] = st.session_state["prediction_history"][:20]
        except requests.RequestException as exc:
            st.error(f"Could not reach the Railway API: {exc}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if prediction is None and st.session_state["prediction_history"]:
        last = st.session_state["prediction_history"][0]
        prediction = {
            "optimal_price": float(last["Predicted Price"]),
            "price_change_pct": ((float(last["Predicted Price"]) - inputs["base_price"]) / max(inputs["base_price"], 1e-6)) * 100.0,
            "strategy": last["Strategy"],
            "recommendation": "Latest session prediction loaded.",
        }

    if prediction is None:
        st.info("Select inputs in the sidebar and click Predict Price to generate live insights.")
        st.stop()

    render_top_metrics(inputs, prediction)
    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    insights_df = render_insights_section(inputs, prediction)
    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    render_business_analytics(inputs, prediction)
    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Competitor Comparison", "Demand vs Inventory", "Monthly Trend", "Feature Importance"])
    with tab1:
        st.plotly_chart(build_competitor_chart(inputs, prediction), use_container_width=True)
    with tab2:
        st.plotly_chart(build_demand_inventory_chart(inputs), use_container_width=True)
    with tab3:
        st.plotly_chart(build_monthly_trend_chart(inputs, prediction), use_container_width=True)
    with tab4:
        st.plotly_chart(build_feature_importance_chart(get_feature_importance()), use_container_width=True)

    st.markdown("<div class='soft-divider'></div>", unsafe_allow_html=True)
    col_report, col_history = st.columns([1, 1])
    with col_report:
        render_report_section(selected_category, inputs, prediction, insights_df)
    with col_history:
        render_prediction_history()

    st.markdown("### Recommendation")
    st.markdown(f"""<div class="info-card glass-card"><div class="small-label">Business Recommendation</div><div style="font-size:1.05rem; font-weight:700; color:white; line-height:1.55;">{prediction.get('recommendation', 'Live recommendation generated.')}</div></div>""", unsafe_allow_html=True)

    st.markdown("### API Health")
    try:
        st.json(api_get("/health"))
    except Exception as exc:
        st.error(f"Unable to fetch backend health: {exc}")


if __name__ == "__main__":
    main()
