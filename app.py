"""
Agmarknet Intelligent Mandi Price & Forecasting System
An Intelligent System to Track Commodity Price Trends Across Mandis Over Time.
Features: Time Series Forecasting (SARIMA), Agmarknet Ingestion, Trend Analysis & Decomposition, Anomaly Detection, Arbitrage Radar.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime
import warnings

# Suppress deprecation and runtime warnings for clean UI
warnings.filterwarnings("ignore")

# Import modular backend
from src.data_loader import AgmarknetDataLoader
from src.trend_analysis import MandiTrendAnalyzer
from src.anomaly_detector import MandiAnomalyDetector
from src.sarima_forecasting import MandiSARIMAForecaster

# ==========================================
# PAGE CONFIGURATION & METADATA
# ==========================================
st.set_page_config(
    page_title="IntelliMandi | Commodity Price Intelligence & SARIMA Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Emerald & Slate Professional Theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2.5rem;
    max-width: 96%;
}

/* Glassmorphic Header Banner */
.hero-card {
    background: linear-gradient(135deg, #064e3b 0%, #047857 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 1.6rem 2rem;
    color: #ffffff;
    box-shadow: 0 10px 25px -5px rgba(6, 78, 59, 0.25);
    margin-bottom: 1.2rem;
    border: 1px solid rgba(255, 255, 255, 0.12);
}

.hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.hero-subtitle {
    font-size: 1.0rem;
    color: #a7f3d0;
    margin-top: 0.4rem;
    font-weight: 400;
}

/* Metric Cards */
.metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    border-left: 4px solid #10b981;
}

.metric-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.metric-val {
    font-size: 1.7rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 0.2rem;
}

.metric-sub {
    font-size: 0.82rem;
    color: #64748b;
    margin-top: 0.2rem;
}

/* Alert Badges */
.badge-critical {
    background: #fee2e2;
    color: #b91c1c;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.75rem;
}

.badge-moderate {
    background: #fef3c7;
    color: #d97706;
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.75rem;
}
</style>
""", unsafe_allow_html=True)

COMMODITY_ICONS = {
    "Tomato": "🍅 Tomato",
    "Onion": "🧅 Onion",
    "Potato": "🥔 Potato",
    "Wheat": "🌾 Wheat",
    "Maize": "🌽 Maize",
    "Mustard": "🌿 Mustard",
    "Cotton": "⚪ Cotton",
    "Soybean": "🌱 Soybean",
    "Paddy (Dhan)": "🌾 Paddy (Dhan)",
    "Dhan (Common)": "🌾 Dhan (Common)"
}

# ==========================================
# DATA LOADING & CACHING
# ==========================================
@st.cache_data(show_spinner=False)
def load_agmarknet_data():
    loader = AgmarknetDataLoader()
    return loader.load_data()

try:
    df_raw = load_agmarknet_data()
    loader = AgmarknetDataLoader()
    loader.df = df_raw
except Exception as e:
    st.error(f"Error loading Agmarknet dataset: {e}")
    st.stop()

# ==========================================
# SIDEBAR CONTROLS & FILTERING
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/wheat.png", width=70)
    st.title("🌾 IntelliMandi")
    st.caption("Agmarknet Intelligent Price Tracking System")
    st.markdown("---")

    # 1. Commodity Selection
    commodities = loader.get_commodities()
    selected_commodity = st.selectbox(
        "Select Commodity",
        options=commodities,
        index=commodities.index("Tomato") if "Tomato" in commodities else 0,
        format_func=lambda x: COMMODITY_ICONS.get(x, x)
    )

    # 2. State & Mandi Filter
    available_states = ["All States"] + loader.get_states(selected_commodity)
    selected_state = st.selectbox("State Filter", options=available_states)

    available_mandis = ["All Mandis"] + loader.get_mandis(selected_commodity)
    selected_mandi = st.selectbox("Mandi (Market Center)", options=available_mandis)

    # 3. Unit Toggle
    unit_choice = st.radio(
        "Price Unit Display",
        options=["₹ / Quintal (100 kg)", "₹ / Kilogram (Retail)"],
        index=0
    )
    is_kg = "Kilogram" in unit_choice
    unit_multiplier = 0.01 if is_kg else 1.0
    unit_label = "₹/kg" if is_kg else "₹/qtl"

    st.markdown("---")
    st.subheader("📅 Date Horizon")

    min_date = df_raw["Date"].min().to_pydatetime()
    max_date = df_raw["Date"].max().to_pydatetime()

    # Preset horizon buttons
    pcol1, pcol2 = st.columns(2)
    if pcol1.button("Last 90 Days", width="stretch"):
        st.session_state["start_date"] = max_date - pd.Timedelta(days=90)
        st.session_state["end_date"] = max_date
    if pcol2.button("Full History", width="stretch"):
        st.session_state["start_date"] = min_date
        st.session_state["end_date"] = max_date

    start_val = st.session_state.get("start_date", min_date)
    end_val = st.session_state.get("end_date", max_date)

    date_range = st.date_input(
        "Custom Date Range",
        value=(start_val, end_val),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_filter, end_filter = date_range
    else:
        start_filter, end_filter = min_date, max_date

# Filter DataFrame
filtered_df = loader.filter_data(
    commodity=selected_commodity,
    mandi=selected_mandi,
    state=selected_state,
    start_date=str(start_filter),
    end_date=str(end_filter)
)

# Apply unit conversions
filtered_df["Display_Modal"] = filtered_df["Modal_Price"] * unit_multiplier
filtered_df["Display_Min"] = filtered_df["Min_Price"] * unit_multiplier
filtered_df["Display_Max"] = filtered_df["Max_Price"] * unit_multiplier
filtered_df["Display_Spread"] = filtered_df["Daily_Spread"] * unit_multiplier

# Extract single continuous daily time series for SARIMA and Trend analysis
daily_series_raw = loader.get_daily_time_series(commodity=selected_commodity, mandi=selected_mandi)
daily_series = daily_series_raw.loc[str(start_filter):str(end_filter)] * unit_multiplier

# ==========================================
# HERO BANNER
# ==========================================
st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">
        {COMMODITY_ICONS.get(selected_commodity, selected_commodity)} Price Intelligence & SARIMA Radar
    </div>
    <div class="hero-subtitle">
        An Intelligent System to Track Commodity Price Trends Across Mandis Over Time • Agmarknet APMC Analytics
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# TOP KPI METRICS ROW
# ==========================================
if not daily_series.empty:
    latest_price = float(daily_series.iloc[-1])
    prev_price = float(daily_series.iloc[-2]) if len(daily_series) > 1 else latest_price
    price_change = latest_price - prev_price
    price_change_pct = (price_change / prev_price * 100) if prev_price > 0 else 0

    avg_price = float(daily_series.mean())
    volatility_cv = float((daily_series.std() / avg_price * 100)) if avg_price > 0 else 0
    active_mandis_cnt = filtered_df["Mandi"].nunique()
    total_arrivals = filtered_df["Arrival_Tonnes"].sum() if "Arrival_Tonnes" in filtered_df.columns else 0

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Latest Modal Price</div>
            <div class="metric-val">₹{latest_price:,.1f}</div>
            <div class="metric-sub" style="color:{'#059669' if price_change >= 0 else '#dc2626'}; font-weight:600;">
                {'▲' if price_change >= 0 else '▼'} ₹{abs(price_change):,.1f} ({price_change_pct:+.1f}%) 1D
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Period Avg Price</div>
            <div class="metric-val">₹{avg_price:,.1f}</div>
            <div class="metric-sub">Range: ₹{daily_series.min():,.0f} – ₹{daily_series.max():,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Volatility Index (CV)</div>
            <div class="metric-val">{volatility_cv:.1f}%</div>
            <div class="metric-sub" style="color:{'#10b981' if volatility_cv < 10 else ('#f59e0b' if volatility_cv < 20 else '#ef4444')}; font-weight:600;">
                {'Low Risk' if volatility_cv < 10 else ('Moderate' if volatility_cv < 20 else 'High Volatility')}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Monitored Mandis</div>
            <div class="metric-val">{active_mandis_cnt}</div>
            <div class="metric-sub">{filtered_df['State'].nunique()} Indian States</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Arrivals Recorded</div>
            <div class="metric-val">{total_arrivals:,.0f}</div>
            <div class="metric-sub">Metric Tonnes</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

# ==========================================
# 6 SPECIALIZED ANALYTICAL TABS
# ==========================================
tabs = st.tabs([
    "📈 Executive Overview",
    "🔍 Trend & Decomposition",
    "⚖️ Cross-Mandi Arbitrage",
    "🚨 Anomaly Detection Hub",
    "🔮 SARIMA Forecasting Studio",
    "📋 Agmarknet Data Explorer"
])

# ----------------------------------------------------
# TAB 1: EXECUTIVE OVERVIEW
# ----------------------------------------------------
with tabs[0]:
    st.subheader(f"📊 Market Price Trajectory — {selected_commodity}")
    st.caption("Daily modal price movement with min-max confidence price bounds across monitored APMC mandis.")

    if not filtered_df.empty:
        fig_overview = px.line(
            filtered_df,
            x="Date",
            y="Display_Modal",
            color="Mandi" if selected_mandi == "All Mandis" else None,
            title=f"Agmarknet Modal Price Trends ({unit_label})",
            labels={"Display_Modal": f"Price ({unit_label})", "Date": "Trading Date"},
            template="plotly_white"
        )
        fig_overview.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=35, b=20),
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9")
        )
        st.plotly_chart(fig_overview, width="stretch")

        # Leaderboard Table on latest reported day
        latest_date = filtered_df["Date"].max()
        latest_slice = filtered_df[filtered_df["Date"] == latest_date]
        
        st.markdown(f"##### 🏆 **Mandi Standings on Latest Reporting Date ({latest_date.strftime('%d %b %Y')})**")
        mandi_board = latest_slice[[
            "Mandi", "State", "Display_Modal", "Display_Min", "Display_Max", "Display_Spread", "Arrival_Tonnes"
        ]].sort_values("Display_Modal", ascending=False).reset_index(drop=True)

        mandi_board.columns = [
            "Mandi Center", "State", f"Modal Price ({unit_label})", f"Min Price ({unit_label})",
            f"Max Price ({unit_label})", f"Spread ({unit_label})", "Arrivals (Tonnes)"
        ]
        st.dataframe(
            mandi_board.style.format({
                f"Modal Price ({unit_label})": "₹{:,.1f}",
                f"Min Price ({unit_label})": "₹{:,.1f}",
                f"Max Price ({unit_label})": "₹{:,.1f}",
                f"Spread ({unit_label})": "₹{:,.1f}",
                "Arrivals (Tonnes)": "{:,.1f}"
            }),
            width="stretch"
        )
    else:
        st.warning("No data found for the current filter criteria.")

# ----------------------------------------------------
# TAB 2: TREND ANALYSIS & SEASONAL DECOMPOSITION
# ----------------------------------------------------
with tabs[1]:
    st.subheader(f"🔍 Time Series Trend & Seasonal Decomposition — {selected_commodity}")
    st.caption("Deconstruct historical prices into underlying Trend, Seasonality, and Irregular Residual components.")

    if len(daily_series) >= 14:
        analyzer = MandiTrendAnalyzer(daily_series)
        tech_df = analyzer.calculate_technical_indicators()

        tcol1, tcol2 = st.columns([2, 1])

        with tcol1:
            st.markdown("##### 📈 **Moving Averages & Bollinger Bands Trend Tunnel**")
            fig_ma = go.Figure()

            # Bollinger band fill
            fig_ma.add_trace(go.Scatter(
                x=tech_df.index, y=tech_df["bollinger_upper"],
                mode="lines", line=dict(color="rgba(16,185,129,0)"), showlegend=False, hoverinfo="skip"
            ))
            fig_ma.add_trace(go.Scatter(
                x=tech_df.index, y=tech_df["bollinger_lower"],
                mode="lines", line=dict(color="rgba(16,185,129,0)"),
                fill="tonexty", fillcolor="rgba(16,185,129,0.12)",
                name="Bollinger Band (20D, ±2σ)"
            ))

            # Actual Price
            fig_ma.add_trace(go.Scatter(
                x=tech_df.index, y=tech_df["price"],
                mode="lines", name="Observed Modal Price",
                line=dict(color="#059669", width=2)
            ))

            # Moving Averages
            fig_ma.add_trace(go.Scatter(
                x=tech_df.index, y=tech_df["sma_7"],
                mode="lines", name="7-Day SMA (Fast)",
                line=dict(color="#d97706", width=1.8, dash="dot")
            ))
            fig_ma.add_trace(go.Scatter(
                x=tech_df.index, y=tech_df["sma_30"],
                mode="lines", name="30-Day SMA (Macro)",
                line=dict(color="#2563eb", width=2)
            ))

            fig_ma.update_layout(
                template="plotly_white", height=380,
                margin=dict(l=20, r=20, t=25, b=20),
                legend=dict(orientation="h", y=1.05, x=0),
                yaxis=dict(title=f"Price ({unit_label})", showgrid=True, gridcolor="#f1f5f9"),
                xaxis=dict(showgrid=True, gridcolor="#f1f5f9")
            )
            st.plotly_chart(fig_ma, width="stretch")

        with tcol2:
            st.markdown("##### 🧪 **Augmented Dickey-Fuller (ADF) Stationarity Test**")
            adf_res = analyzer.check_stationarity_adf()

            stat_badge = "🟢 Stationary (p < 0.05)" if adf_res["is_stationary"] else "🔴 Non-Stationary (p ≥ 0.05)"
            st.markdown(f"""
            <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px; padding:1.2rem;">
                <div style="font-weight:700; font-size:1.1rem; color:#0f172a; margin-bottom:0.5rem;">
                    Status: {stat_badge}
                </div>
                <p style="font-size:0.88rem; color:#334155; margin:0.3rem 0;">
                    <b>ADF Test Statistic:</b> {adf_res['adf_statistic']:.4f}
                </p>
                <p style="font-size:0.88rem; color:#334155; margin:0.3rem 0;">
                    <b>p-Value:</b> {adf_res['p_value']:.4e}
                </p>
                <p style="font-size:0.88rem; color:#334155; margin:0.3rem 0;">
                    <b>Critical Value (5%):</b> {adf_res['critical_values'].get('5%', -2.86):.4f}
                </p>
                <div style="margin-top:0.8rem; padding-top:0.8rem; border-top:1px dashed #cbd5e1; font-size:0.85rem; color:#047857;">
                    <b>SARIMA Advisory:</b> {adf_res['recommendation']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 4-pane STL Classical Decomposition
        st.markdown("##### 🔬 **STL Seasonal Decomposition (Observed = Trend + Seasonal + Residual)**")
        decomp_period = st.select_slider(
            "Select Seasonal Period (Days)",
            options=[7, 14, 30],
            value=7,
            format_func=lambda x: f"{x} Days ({'Weekly Procurement Cycle' if x==7 else ('Bi-Weekly' if x==14 else 'Monthly')})"
        )

        decomp = analyzer.decompose_time_series(period=decomp_period, model="additive")
        if decomp:
            fig_decomp = make_subplots(
                rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                subplot_titles=(
                    f"Observed Price ({unit_label})",
                    f"Underlying Trend Component ({unit_label})",
                    f"Seasonal Cycle ({decomp_period}-Day Repeat)",
                    "Irregular Residuals (Unsystematic Shock)"
                )
            )

            fig_decomp.add_trace(go.Scatter(x=decomp["observed"].index, y=decomp["observed"].values, line=dict(color="#0f172a")), row=1, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp["trend"].index, y=decomp["trend"].values, line=dict(color="#059669", width=2)), row=2, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp["seasonal"].index, y=decomp["seasonal"].values, line=dict(color="#2563eb")), row=3, col=1)
            fig_decomp.add_trace(go.Scatter(x=decomp["residual"].index, y=decomp["residual"].values, mode="markers", marker=dict(color="#dc2626", size=3.5)), row=4, col=1)

            fig_decomp.update_layout(height=600, template="plotly_white", showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_decomp, width="stretch")
        else:
            st.info("Insufficient observations in current window for seasonal decomposition.")
    else:
        st.warning("Need at least 14 days of data to perform trend decomposition.")

# ----------------------------------------------------
# TAB 3: CROSS-MANDI ARBITRAGE RADAR
# ----------------------------------------------------
with tabs[2]:
    st.subheader("⚖️ Cross-Mandi Price Dispersion & Arbitrage Radar")
    st.caption("Identify inter-mandi spatial spreads, arbitrage margins between producing and consuming centers.")

    arb = MandiTrendAnalyzer.analyze_cross_mandi_arbitrage(filtered_df)

    if arb and arb.get("has_arbitrage"):
        acol1, acol2 = st.columns([1.3, 1])

        with acol1:
            st.markdown(f"""
            <div style="background:#ffffff; border:1px solid #cbd5e1; border-radius:14px; padding:1.4rem; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05);">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-weight:700; font-size:1.1rem; color:#0f172a;">⚡ Real-Time Arbitrage Opportunity</span>
                    <span style="background:#ecfdf5; color:#059669; font-weight:700; font-size:0.75rem; padding:0.25rem 0.6rem; border-radius:9999px;">Active Spread</span>
                </div>
                <div style="margin-top:1rem; display:grid; grid-template-columns: 1fr 1fr; gap:1rem;">
                    <div style="background:#f0fdf4; padding:0.8rem 1rem; border-radius:10px; border-left:3px solid #16a34a;">
                        <div style="font-size:0.75rem; color:#166534; font-weight:700; text-transform:uppercase;">Highest Selling Market</div>
                        <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin-top:0.2rem;">{arb['highest_mandi']}</div>
                        <div style="font-size:0.85rem; color:#15803d; font-weight:600;">₹{arb['highest_price'] * unit_multiplier:,.1f} {unit_label} ({arb['highest_state']})</div>
                    </div>
                    <div style="background:#fef2f2; padding:0.8rem 1rem; border-radius:10px; border-left:3px solid #dc2626;">
                        <div style="font-size:0.75rem; color:#991b1b; font-weight:700; text-transform:uppercase;">Lowest Procurement Market</div>
                        <div style="font-size:1.25rem; font-weight:800; color:#0f172a; margin-top:0.2rem;">{arb['lowest_mandi']}</div>
                        <div style="font-size:0.85rem; color:#b91c1c; font-weight:600;">₹{arb['lowest_price'] * unit_multiplier:,.1f} {unit_label} ({arb['lowest_state']})</div>
                    </div>
                </div>
                <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px dashed #cbd5e1; display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase;">Gross Arbitrage Margin</div>
                        <div style="font-size:1.5rem; font-weight:800; color:#d97706;">₹{arb['gross_spread_rs'] * unit_multiplier:,.1f} {unit_label}</div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.75rem; color:#64748b; font-weight:600; text-transform:uppercase;">Potential Premium</div>
                        <div style="font-size:1.5rem; font-weight:800; color:#059669;">+{arb['spread_percentage']:.1f}%</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with acol2:
            st.markdown("##### 📍 **Mandi Price Dispersion (Latest Trading Day)**")
            summary_df = pd.DataFrame(arb["mandi_summary"])
            summary_df["Display_Price"] = summary_df["Modal_Price"] * unit_multiplier
            
            fig_arb_bar = px.bar(
                summary_df,
                x="Display_Price",
                y="Mandi",
                orientation="h",
                color="Display_Price",
                color_continuous_scale="Viridis",
                labels={"Display_Price": f"Price ({unit_label})"},
                template="plotly_white",
                text_auto=".1f"
            )
            fig_arb_bar.update_layout(height=280, margin=dict(l=20, r=20, t=10, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_arb_bar, width="stretch")

        # Cross-mandi spatial boxplot
        st.markdown("##### 📦 **State-Wise Price Distribution Over Selected Timeline**")
        fig_box = px.box(
            filtered_df,
            x="State",
            y="Display_Modal",
            color="State",
            points="outliers",
            template="plotly_white",
            labels={"Display_Modal": f"Modal Price ({unit_label})"}
        )
        fig_box.update_layout(height=340, margin=dict(l=20, r=20, t=10, b=20), showlegend=False)
        st.plotly_chart(fig_box, width="stretch")
    else:
        st.info("Select 'All Mandis' or broaden the date range to view inter-mandi arbitrage comparisons.")

# ----------------------------------------------------
# TAB 4: ANOMALY DETECTION HUB
# ----------------------------------------------------
with tabs[3]:
    st.subheader(f"🚨 Anomaly Detection Radar — {selected_commodity}")
    st.caption("Detect sudden price spikes, unseasonal market crashes, supply deficit shocks, and statistical outliers.")

    anomaly_detector = MandiAnomalyDetector(filtered_df)
    
    # Anomaly sensitivity controls
    acol_s1, acol_s2 = st.columns([1, 1])
    with acol_s1:
        z_thresh = st.slider("Z-Score Sensitivity Threshold (Standard Deviations)", 1.5, 4.0, 2.5, 0.1)
    with acol_s2:
        iso_contam = st.slider("Isolation Forest Contamination Rate", 0.01, 0.10, 0.04, 0.01)

    unified_anomalies = anomaly_detector.run_unified_anomaly_detection(
        z_threshold=z_thresh,
        contamination=iso_contam
    )

    # Anomaly summary badges
    total_anomalies = unified_anomalies["is_anomaly"].sum()
    critical_shocks = (unified_anomalies["severity"] == "Critical Price Shock").sum()
    moderate_devs = (unified_anomalies["severity"] == "Moderate Deviation").sum()

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Total Anomalies Detected", f"{total_anomalies} Events")
    mcol2.metric("Critical Price Shocks", f"{critical_shocks} Critical", delta_color="inverse")
    mcol3.metric("Moderate Deviations", f"{moderate_devs} Warnings")
    mcol4.metric("Market Normalcy Rate", f"{((len(unified_anomalies) - total_anomalies)/len(unified_anomalies)*100):.1f}%")

    # Time series chart with anomaly markers
    st.markdown("##### 📍 **Time Series Anomaly Overlay**")
    fig_anom = go.Figure()

    # Normal line
    fig_anom.add_trace(go.Scatter(
        x=unified_anomalies["Date"],
        y=unified_anomalies["Modal_Price"] * unit_multiplier,
        mode="lines",
        name="Modal Price",
        line=dict(color="#64748b", width=1.8)
    ))

    # Anomaly markers
    anom_points = unified_anomalies[unified_anomalies["is_anomaly"]]
    if not anom_points.empty:
        # Critical markers
        crit_points = anom_points[anom_points["severity"] == "Critical Price Shock"]
        if not crit_points.empty:
            fig_anom.add_trace(go.Scatter(
                x=crit_points["Date"],
                y=crit_points["Modal_Price"] * unit_multiplier,
                mode="markers",
                name="Critical Shock",
                marker=dict(color="#dc2626", size=10, symbol="x-thin", line=dict(width=3, color="#dc2626")),
                text=crit_points["anomaly_reason"],
                hoverinfo="x+y+text"
            ))

        # Moderate markers
        mod_points = anom_points[anom_points["severity"] == "Moderate Deviation"]
        if not mod_points.empty:
            fig_anom.add_trace(go.Scatter(
                x=mod_points["Date"],
                y=mod_points["Modal_Price"] * unit_multiplier,
                mode="markers",
                name="Moderate Anomaly",
                marker=dict(color="#d97706", size=8, symbol="circle"),
                text=mod_points["anomaly_reason"],
                hoverinfo="x+y+text"
            ))

    fig_anom.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        yaxis=dict(title=f"Price ({unit_label})", showgrid=True, gridcolor="#f1f5f9"),
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        legend=dict(orientation="h", y=1.05, x=0)
    )
    st.plotly_chart(fig_anom, width="stretch")

    # Anomaly Incident Log Table
    st.markdown("##### 📋 **Anomaly Incident Audit Log**")
    if total_anomalies > 0:
        anom_log = anom_points[[
            "Date", "Mandi", "Modal_Price", "z_score", "severity", "anomaly_reason"
        ]].copy()
        anom_log["Display_Modal"] = anom_log["Modal_Price"] * unit_multiplier
        anom_log = anom_log.drop(columns=["Modal_Price"]).sort_values("Date", ascending=False)
        anom_log.columns = ["Date", "Mandi Center", "Z-Score", "Severity Tier", "Root Cause & Drivers", f"Price ({unit_label})"]

        st.dataframe(
            anom_log.style.format({
                f"Price ({unit_label})": "₹{:,.1f}",
                "Z-Score": "{:+.2f}σ"
            }),
            width="stretch"
        )
    else:
        st.success("No anomalies detected within current parameter thresholds.")

# ----------------------------------------------------
# TAB 5: SARIMA FORECASTING STUDIO
# ----------------------------------------------------
with tabs[4]:
    st.subheader(f"🔮 SARIMA Time Series Forecasting Studio — {selected_commodity}")
    st.caption("Seasonal AutoRegressive Integrated Moving Average SARIMA(p, d, q) × (P, D, Q)s predictive engine.")

    if len(daily_series) >= 20:
        forecaster = MandiSARIMAForecaster(daily_series, seasonal_period=7)

        # Control Columns
        scol1, scol2, scol3 = st.columns([1.2, 1.2, 1.2])

        with scol1:
            st.markdown("##### ⚙️ Non-Seasonal Order (p, d, q)")
            p_val = st.number_input("AR Order (p)", min_value=0, max_value=3, value=1)
            d_val = st.number_input("Differencing Order (d)", min_value=0, max_value=2, value=1)
            q_val = st.number_input("MA Order (q)", min_value=0, max_value=3, value=1)

        with scol2:
            st.markdown("##### 🔁 Seasonal Order (P, D, Q)s")
            P_val = st.number_input("Seasonal AR (P)", min_value=0, max_value=2, value=1)
            D_val = st.number_input("Seasonal Diff (D)", min_value=0, max_value=1, value=0)
            Q_val = st.number_input("Seasonal MA (Q)", min_value=0, max_value=2, value=1)
            s_val = st.selectbox("Seasonality (s)", options=[7, 14, 30], index=0, format_func=lambda x: f"s={x} (Weekly)")

        with scol3:
            st.markdown("##### 🎯 Forecast Horizon")
            forecast_horizon = st.select_slider(
                "Days to Forecast Ahead",
                options=[7, 14, 21, 30, 45, 60],
                value=30
            )
            auto_btn = st.button("⚡ Auto-Fit Optimal SARIMA Order", width="stretch")

        user_order = (int(p_val), int(d_val), int(q_val))
        user_seasonal_order = (int(P_val), int(D_val), int(Q_val), int(s_val))

        if auto_btn:
            with st.spinner("Executing AIC-minimizing Grid Search..."):
                best_order, best_seasonal = forecaster.auto_fit_best_order()
                st.success(f"Best SARIMA Order Identified: {best_order} × {best_seasonal}")
                user_order = best_order
                user_seasonal_order = best_seasonal

        # 1. Backtesting Validation
        with st.spinner("Computing SARIMA Validation & Out-of-Sample Forecast..."):
            val_results = forecaster.evaluate_train_test(
                order=user_order,
                seasonal_order=user_seasonal_order,
                test_size=min(28, max(7, int(len(daily_series) * 0.15)))
            )

            # 2. Forward Out-of-Sample Projection
            forecast_results = forecaster.forecast_future(
                steps=forecast_horizon,
                order=user_order,
                seasonal_order=user_seasonal_order
            )

        # Quantitative Metrics Bar
        q1, q2, q3, q4, q5 = st.columns(5)
        q1.metric("MAE (Mean Abs Error)", f"₹{val_results['mae']:,.2f}")
        q2.metric("RMSE (Root Mean Sq)", f"₹{val_results['rmse']:,.2f}")
        q3.metric("MAPE (Percentage Error)", f"{val_results['mape']:.2f}%")
        q4.metric("Directional Accuracy", f"{val_results['directional_accuracy_pct']:.1f}%")
        q5.metric("Model AIC / BIC", f"{val_results['aic']:,.0f} / {val_results['bic']:,.0f}")

        # Forecast Chart
        st.markdown(f"##### 📈 **SARIMA {user_order}×{user_seasonal_order} Forward Projection ({forecast_horizon} Days)**")
        fig_sarima = go.Figure()

        # Historical Actuals
        fig_sarima.add_trace(go.Scatter(
            x=daily_series.index,
            y=daily_series.values,
            mode="lines",
            name="Historical Actual Price",
            line=dict(color="#059669", width=2.2)
        ))

        # 95% Confidence Interval Band
        f_dates = list(forecast_results["future_dates"])
        f_upper = list(forecast_results["ci_95_upper"])
        f_lower = list(forecast_results["ci_95_lower"])
        
        # Bridge to last observed point
        bridge_x = [daily_series.index[-1]] + f_dates
        bridge_upper = [daily_series.iloc[-1]] + f_upper
        bridge_lower = [daily_series.iloc[-1]] + f_lower
        bridge_pred = [daily_series.iloc[-1]] + list(forecast_results["forecast_mean"])

        fig_sarima.add_trace(go.Scatter(
            x=bridge_x + bridge_x[::-1],
            y=bridge_upper + bridge_lower[::-1],
            fill="toself",
            fillcolor="rgba(16, 185, 129, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Cone",
            hoverinfo="skip"
        ))

        # Forecast Mean
        fig_sarima.add_trace(go.Scatter(
            x=bridge_x,
            y=bridge_pred,
            mode="lines+markers",
            name=f"SARIMA Forecast ({forecast_horizon}D)",
            line=dict(color="#d97706", width=2.8, dash="dash"),
            marker=dict(size=5, symbol="diamond")
        ))

        fig_sarima.update_layout(
            template="plotly_white",
            height=440,
            margin=dict(l=20, r=20, t=25, b=20),
            yaxis=dict(title=f"Price ({unit_label})", showgrid=True, gridcolor="#f1f5f9"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            legend=dict(orientation="h", y=1.05, x=0)
        )
        st.plotly_chart(fig_sarima, width="stretch")

        # Residual Diagnostics
        st.markdown("##### 🔬 **SARIMA Residual Diagnostics (Checking White Noise Assumptions)**")
        rcol1, rcol2 = st.columns([1.5, 1])

        with rcol1:
            if not forecast_results["residuals"].empty:
                res_df = pd.DataFrame({"Residual": forecast_results["residuals"]})
                fig_resid = px.line(
                    res_df,
                    y="Residual",
                    title="Model Residuals Over Time (Target: Mean Zero & Stationary)",
                    template="plotly_white"
                )
                fig_resid.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_resid, width="stretch")

        with rcol2:
            if not forecast_results["residuals"].empty:
                fig_hist = px.histogram(
                    forecast_results["residuals"],
                    nbins=25,
                    title="Residual Distribution (Target: Normal Curve)",
                    template="plotly_white",
                    color_discrete_sequence=["#10b981"]
                )
                fig_hist.update_layout(height=260, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
                st.plotly_chart(fig_hist, width="stretch")

    else:
        st.warning("Insufficient data points in current window to train SARIMA model.")

# ----------------------------------------------------
# TAB 6: DATA EXPLORER & EXPORT
# ----------------------------------------------------
with tabs[5]:
    st.subheader("📋 Agmarknet Mandi Price Records Explorer")
    st.caption("Search, filter, inspect, and export cleaned Agmarknet market arrival and price transactions.")

    if not filtered_df.empty:
        col_s1, col_s2 = st.columns([2, 1])
        with col_s1:
            search_query = st.text_input("Search Mandi, State, or Variety...", "")
        with col_s2:
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Filtered Data as CSV",
                data=csv_data,
                file_name=f"agmarknet_{selected_commodity.lower()}_prices.csv",
                mime="text/csv",
                width="stretch"
            )

        display_df = filtered_df.copy()
        if search_query:
            mask = (
                display_df["Mandi"].str.contains(search_query, case=False, na=False) |
                display_df["State"].str.contains(search_query, case=False, na=False)
            )
            display_df = display_df[mask]

        st.dataframe(
            display_df[[
                "Date", "State", "Mandi", "Commodity", "Variety", "Arrival_Tonnes",
                "Display_Min", "Display_Max", "Display_Modal", "Display_Spread"
            ]].style.format({
                "Arrival_Tonnes": "{:,.1f}",
                "Display_Min": "₹{:,.1f}",
                "Display_Max": "₹{:,.1f}",
                "Display_Modal": "₹{:,.1f}",
                "Display_Spread": "₹{:,.1f}"
            }),
            width="stretch",
            height=450
        )
    else:
        st.info("No records matching the filter criteria.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem; padding: 0.5rem 0;">
    🌾 <b>Intelligent Mandi Price & SARIMA Forecasting System</b> | Built for Agmarknet Agricultural Market Decision Support | Antigravity AI
</div>
""", unsafe_allow_html=True)
