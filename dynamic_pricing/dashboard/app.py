import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="AI Dynamic Pricing Engine", page_icon="💰", layout="wide")

API_URL = "http://localhost:8000"

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 28px; font-weight: bold; color: #00cc96; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Dynamic Pricing Optimization Engine")
st.markdown("**Real-time price recommendations powered by Gradient Boosting ML**")
st.divider()

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Product Configuration")
    category = st.selectbox("Category", ["Electronics", "Clothing", "Groceries", "Books", "Sports", "Home & Garden"])

    st.subheader("💰 Pricing")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        base_price = st.slider("Your Current Price ($)", 1.0, 2000.0, 100.0, step=0.5)
    with col_b:
        base_price = st.number_input("", min_value=1.0, max_value=2000.0,
                                      value=base_price, step=0.5, key="bp_input", label_visibility="hidden")

    col_c, col_d = st.columns([2, 1])
    with col_c:
        competitor_1 = st.slider("Competitor A Price ($)", 1.0, 2000.0, 95.0, step=0.5)
    with col_d:
        competitor_1 = st.number_input("", min_value=1.0, max_value=2000.0,
                                        value=competitor_1, step=0.5, key="c1_input", label_visibility="hidden")

    col_e, col_f = st.columns([2, 1])
    with col_e:
        competitor_2 = st.slider("Competitor B Price ($)", 1.0, 2000.0, 98.0, step=0.5)
    with col_f:
        competitor_2 = st.number_input("", min_value=1.0, max_value=2000.0,
                                        value=competitor_2, step=0.5, key="c2_input", label_visibility="hidden")

    st.subheader("📦 Inventory & Demand")

    col_g, col_h = st.columns([2, 1])
    with col_g:
        inventory = st.slider("Inventory Level (units)", 0, 500, 100)
    with col_h:
        inventory = st.number_input("", min_value=0, max_value=500,
                                     value=inventory, step=1, key="inv_input", label_visibility="hidden")

    col_i, col_j = st.columns([2, 1])
    with col_i:
        demand_score = st.slider("Demand Score (0=Low, 1=High)", 0.0, 1.0, 0.6, step=0.01)
    with col_j:
        demand_score = st.number_input("", min_value=0.0, max_value=1.0,
                                        value=demand_score, step=0.01, key="dem_input", label_visibility="hidden")

    col_k, col_l = st.columns([2, 1])
    with col_k:
        rating = st.slider("Customer Rating", 1.0, 5.0, 4.2, step=0.1)
    with col_l:
        rating = st.number_input("", min_value=1.0, max_value=5.0,
                                  value=rating, step=0.1, key="rat_input", label_visibility="hidden")

    st.subheader("📅 Timing")
    day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    day = st.selectbox("Day of Week", day_names, index=datetime.now().weekday())
    month = st.selectbox("Month", range(1, 13), index=datetime.now().month - 1)
    predict_btn = st.button("🚀 Get Optimal Price", use_container_width=True)

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💰 Price Optimizer", "📊 Revenue Analysis", "🏪 Competitor Intel", "📈 Weekly Simulation"])

# ── Tab 1: Optimizer ──────────────────────────────────────
with tab1:
    if predict_btn:
        day_num = day_names.index(day)
        payload = {
            "category": category, "base_price": base_price,
            "competitor_price_1": competitor_1, "competitor_price_2": competitor_2,
            "inventory_level": inventory, "demand_score": demand_score,
            "day_of_week": day_num, "month": month, "customer_rating": rating
        }
        try:
            resp = requests.post(f"{API_URL}/predict-price", json=payload, timeout=5)
            result = resp.json()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Price", f"${base_price:.2f}")
            c2.metric("AI Optimal Price", f"${result['optimal_price']:.2f}", delta=f"{result['price_change_pct']:+.1f}%")
            est_uplift = abs(result['price_change_pct']) * base_price * demand_score
            c3.metric("Est. Revenue Uplift", f"${est_uplift:.0f}/day")
            c4.metric("Strategy", result['strategy'])

            st.success(f"**AI Recommendation:** {result['recommendation']}")

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=result['optimal_price'],
                delta={'reference': base_price, 'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [0, base_price * 1.6]},
                    'bar': {'color': "#00cc96"},
                    'steps': [
                        {'range': [0, base_price * 0.8], 'color': "#ff4444"},
                        {'range': [base_price * 0.8, base_price * 1.1], 'color': "#ffaa00"},
                        {'range': [base_price * 1.1, base_price * 1.6], 'color': "#2ecc71"}
                    ],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'value': base_price}
                },
                title={'text': "Recommended Price ($)"}
            ))
            fig.update_layout(height=320, paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig, use_container_width=True)

            # Feature importance
            try:
                fi_resp = requests.get(f"{API_URL}/feature-importance").json()
                fi_df = pd.DataFrame(fi_resp.items(), columns=['Feature', 'Importance']).sort_values('Importance', ascending=True)
                fig2 = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
                              title="🔍 What Drives the Price Decision",
                              color='Importance', color_continuous_scale='viridis')
                fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=350)
                st.plotly_chart(fig2, use_container_width=True)
            except:
                pass

        except Exception as e:
            st.error(f"⚠️ Could not reach API. Make sure FastAPI is running: uvicorn api.main:app --reload")
    else:
        st.info("👈 Configure your product in the sidebar and click **Get Optimal Price**")
        col1, col2, col3 = st.columns(3)
        col1.markdown("**🔍 Input Factors**\n- Category & base price\n- Competitor prices\n- Inventory levels\n- Customer demand score\n- Season & day of week")
        col2.markdown("**🤖 AI Model**\n- Gradient Boosting ML\n- Trained on 5,000 samples\n- Price elasticity analysis\n- Demand-revenue optimization\n- Real-time inference (<50ms)")
        col3.markdown("**💡 Output**\n- Optimal price recommendation\n- Revenue uplift estimate\n- Strategy classification\n- Feature importance\n- Action plan")

# ── Tab 2: Revenue Analysis ───────────────────────────────
with tab2:
    st.subheader("Revenue vs Price Curve")
    price_range = np.linspace(base_price * 0.4, base_price * 2.0, 80)
    elasticity = -1.5 if demand_score > 0.5 else -2.2
    revenues = [p * max(100 * (p / base_price) ** elasticity * demand_score, 0) for p in price_range]
    optimal_idx = int(np.argmax(revenues))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=price_range, y=revenues, mode='lines', name='Revenue',
                             line=dict(color='#00cc96', width=3), fill='tozeroy', fillcolor='rgba(0,204,150,0.1)'))
    fig.add_vline(x=base_price, line_dash="dash", line_color="#4488ff", annotation_text="Your Price")
    fig.add_vline(x=price_range[optimal_idx], line_dash="dash", line_color="#ff6600",
                  annotation_text=f"Optimal ${price_range[optimal_idx]:.2f}")
    fig.update_layout(title="Revenue Optimization Curve — Find Your Sweet Spot",
                      xaxis_title="Price ($)", yaxis_title="Revenue ($)",
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0.1)',
                      font_color='white', height=420)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        demands = [max(100 * (p / base_price) ** elasticity * demand_score, 0) for p in price_range]
        fig2 = go.Figure(go.Scatter(x=price_range, y=demands, mode='lines',
                                     line=dict(color='#f093fb', width=2)))
        fig2.update_layout(title="Demand Curve", xaxis_title="Price ($)", yaxis_title="Units Sold",
                            paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        cats = ["Electronics", "Clothing", "Groceries", "Books", "Sports", "Home & Garden"]
        avg_rev = [1200, 180, 55, 45, 320, 210]
        fig3 = px.bar(x=cats, y=avg_rev, title="Avg Revenue by Category",
                      color=avg_rev, color_continuous_scale='plasma')
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=300)
        st.plotly_chart(fig3, use_container_width=True)

# ── Tab 3: Competitor Intel ───────────────────────────────
with tab3:
    st.subheader("Competitive Price Positioning")
    col1, col2 = st.columns([1, 2])

    with col1:
        market_avg = (base_price + competitor_1 + competitor_2) / 3
        for name, price in [("🔵 You", base_price), ("🔴 Competitor A", competitor_1),
                             ("🟠 Competitor B", competitor_2), ("⚪ Market Avg", market_avg)]:
            diff = ((price - base_price) / base_price * 100) if "You" not in name else 0
            st.metric(name, f"${price:.2f}", delta=f"{diff:+.1f}%" if "You" not in name else None)

        if base_price < min(competitor_1, competitor_2):
            st.success("✅ You are the cheapest — room to increase price!")
        elif base_price > max(competitor_1, competitor_2):
            st.warning("⚠️ You are the most expensive — consider reducing price")
        else:
            st.info("ℹ️ You are mid-market — optimize based on demand")

    with col2:
        metrics = ['Price Competitiveness', 'Customer Rating', 'Demand', 'Inventory', 'Market Share', 'Revenue Score']
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=[1 - (base_price/market_avg - 1), rating/5, demand_score, inventory/500, 0.35, 0.4],
                                      theta=metrics, name='You', fill='toself', fillcolor='rgba(0,204,150,0.3)'))
        fig.add_trace(go.Scatterpolar(r=[1 - (competitor_1/market_avg - 1), 0.75, 0.55, 0.6, 0.40, 0.45],
                                      theta=metrics, name='Competitor A', fill='toself', fillcolor='rgba(244,67,54,0.2)'))
        fig.add_trace(go.Scatterpolar(r=[1 - (competitor_2/market_avg - 1), 0.70, 0.50, 0.45, 0.25, 0.3],
                                      theta=metrics, name='Competitor B', fill='toself', fillcolor='rgba(255,165,0,0.2)'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                          title="Competitive Radar Analysis", paper_bgcolor='rgba(0,0,0,0)',
                          font_color='white', height=420)
        st.plotly_chart(fig, use_container_width=True)

# ── Tab 4: Weekly Simulation ─────────────────────────────
with tab4:
    st.subheader("7-Day Dynamic vs Static Pricing Simulation")
    days_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    demand_mult = [0.80, 0.85, 0.90, 0.92, 1.00, 1.20, 1.15]
    np.random.seed(42)
    dynamic_prices = [base_price * dm * (1 + np.random.uniform(-0.02, 0.06)) for dm in demand_mult]
    static_prices = [base_price] * 7
    dynamic_rev = [p * dm * 100 for p, dm in zip(dynamic_prices, demand_mult)]
    static_rev = [base_price * dm * 100 for dm in demand_mult]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=days_labels, y=dynamic_prices, name='Dynamic Price (AI)', marker_color='#00cc96'))
    fig.add_trace(go.Scatter(x=days_labels, y=static_prices, mode='lines+markers',
                             name='Static Price', line=dict(color='red', dash='dash', width=2)))
    fig.update_layout(title="Price Strategy Comparison (Dynamic vs Static)",
                      yaxis_title="Price ($)", paper_bgcolor='rgba(0,0,0,0)',
                      font_color='white', height=380)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    total_static = sum(static_rev)
    total_dynamic = sum(dynamic_rev)
    uplift_pct = ((total_dynamic - total_static) / total_static) * 100
    c1.metric("Static Revenue (7 days)", f"${total_static:,.0f}")
    c2.metric("Dynamic Revenue (7 days)", f"${total_dynamic:,.0f}", delta=f"+${total_dynamic-total_static:,.0f}")
    c3.metric("Total Revenue Uplift", f"{uplift_pct:.1f}%", delta="vs static pricing 🎯")

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=days_labels, y=dynamic_rev, mode='lines+markers',
                              name='Dynamic Revenue', line=dict(color='#00cc96', width=3), fill='tozeroy',
                              fillcolor='rgba(0,204,150,0.15)'))
    fig2.add_trace(go.Scatter(x=days_labels, y=static_rev, mode='lines+markers',
                              name='Static Revenue', line=dict(color='red', dash='dash', width=2)))
    fig2.update_layout(title="Revenue Comparison", yaxis_title="Revenue ($)",
                       paper_bgcolor='rgba(0,0,0,0)', font_color='white', height=320)
    st.plotly_chart(fig2, use_container_width=True)