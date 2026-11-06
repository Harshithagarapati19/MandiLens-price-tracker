# 🌾 An Intelligent System to Track Commodity Price Trends Across Mandis Over Time
> **High-Fidelity Agmarknet Price Intelligence, Seasonal Trend Decomposition, Anomaly Detection & SARIMA Time Series Forecasting Dashboard**

---

## 📌 Executive Summary & Problem Context
In India, agricultural commodity prices fluctuate significantly across wholesale markets (**APMC Mandis**) due to seasonal harvest rhythms, transport bottlenecks, monsoon anomalies, and local supply-demand shocks. Farmers, procurement officers, and agri-traders often suffer from informational asymmetry, leading to suboptimal selling decisions and lost arbitrage margins.

This project delivers **an end-to-end intelligent analytical decision-support system** that ingests multi-mandi daily **Agmarknet** transactions, decomposes underlying time series trends, flags market anomalies (unseasonal price spikes or post-harvest glut crashes), computes spatial inter-mandi arbitrage spreads, and generates multi-step forward predictive forecasts using **Seasonal AutoRegressive Integrated Moving Average (SARIMA)** models with 95% confidence intervals.

---

## 🏗️ System Architecture & Workflow

```
[ Agmarknet Data Ingestion ]
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│                 Data Harmonization & Cleaning               │
│ (Daily Resampling, Linear Interpolation, Unit Conversions)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌─────────────────┐
│   Trend &    │   │   Anomaly    │   │     SARIMA      │
│ Seasonality  │   │  Detection   │   │   Forecasting   │
│  (STL, ADF)  │   │ (Z, IQR, IF) │   │ (p,d,q)(P,D,Q)s │
└──────┬───────┘   └──────┬───────┘   └────────┬────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│            Interactive Intelligence Dashboards              │
│   • Streamlit Web Application (app.py)                     │
│   • Standalone Client-Side App (index.html)                 │
│   • Jupyter / Google Colab Research Notebook                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔬 Core Methodologies & Mathematical Foundations

### 1. Time Series Stationarity & Augmented Dickey-Fuller (ADF) Test
To ensure proper specification of the differencing order $d$ in SARIMA, the series $Y_t$ is evaluated under the null hypothesis of a unit root:
$$\Delta Y_t = \alpha + \beta t + \gamma Y_{t-1} + \sum_{i=1}^{k} \delta_i \Delta Y_{t-i} + \epsilon_t$$
- If $p\text{-value} < 0.05$, the series is stationary ($d=0$).
- If $p\text{-value} \ge 0.05$, first-order differencing ($d=1$) is automatically recommended.

### 2. Classical & STL Seasonal Decomposition
Deconstructs observed commodity price $Y_t$ into deterministic and stochastic components:
$$Y_t = T_t + S_t + R_t$$
Where:
- $T_t$: Underlying macro trend
- $S_t$: Cyclical seasonal repeat pattern ($s=7$ for weekly APMC procurement rhythm)
- $R_t$: Irregular residual shocks (unmodeled market disturbances)

### 3. Multi-Engine Anomaly Detection
Combines four complementary anomaly detection mechanisms:
1. **Dynamic Rolling Z-Score:**
   $$Z_t = \frac{Y_t - \mu_{w}(t)}{\sigma_{w}(t)}, \quad \text{Flagged if } |Z_t| \ge 2.5\sigma$$
2. **Interquartile Range (IQR) Extreme Value Boundary:**
   $$\text{Thresholds} = [Q_1 - 1.5 \times IQR, \quad Q_3 + 1.5 \times IQR]$$
3. **Supply-Shock Multivariate Radar:**
   Cross-analyzes daily arrival volume ($A_t$) against modal price ($Y_t$):
   - *Supply Deficit Spike:* $A_t \le P_{20}(A)$ and $Y_t \ge P_{80}(Y)$
   - *Market Glut Crash:* $A_t \ge P_{80}(A)$ and $Y_t \le P_{20}(Y)$
4. **Unsupervised Isolation Forest:**
   Isolates multidimensional outlier vectors $[\Delta Y_t, \sigma_{7}(t), \Delta A_t, \text{Spread}_t]$.

### 4. SARIMA Time Series Forecasting
The Seasonal AutoRegressive Integrated Moving Average model captures both intra-week procurement seasonality and non-stationary price dynamics:
$$\Phi_P(B^s)\phi_p(B)(1-B)^d(1-B^s)^D Y_t = \Theta_Q(B^s)\theta_q(B)\epsilon_t$$
- **Non-seasonal parameters:** $(p, d, q)$
- **Seasonal parameters:** $(P, D, Q)_s$ with seasonal lag $s=7$ (or $s=30$)
- **Validation Metrics:**
  - $\text{MAE} = \frac{1}{n}\sum |y_i - \hat{y}_i|$
  - $\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}$
  - $\text{MAPE} = \frac{1}{n}\sum \left|\frac{y_i - \hat{y}_i}{y_i}\right| \times 100\%$
  - **Directional Accuracy:** Percentage of correct up/down price movement sign predictions.

---

## 🎛️ Dashboard Modules & Analytical Tabs

| Tab | Feature Set | Analytical Value |
|---|---|---|
| **1. Executive Overview** | Daily modal price curve, 24-hour delta, CV% risk rating, and live leaderboard | Instant market snapshot for producers and traders |
| **2. Trend & Decomposition** | Moving averages (SMA 7/14/30), Bollinger bands, ADF test card, 4-pane STL decomposition | Distinguishes permanent shifts from recurring harvest cycles |
| **3. Cross-Mandi Arbitrage** | Highest vs lowest quoting mandis, gross price spread, and state-level dispersion boxplots | Uncovers geographical price disparities and trade arbitrage |
| **4. Anomaly Detection Hub** | Visual scatter overlays on price charts, severity tiers (Critical / Moderate), root-cause logs | Early warning system against price-gouging and supply collapses |
| **5. SARIMA Forecasting Studio** | Interactive order configuration, auto-fit grid search, 7-60 day forecast horizon, 95% confidence cones, error diagnostics | Accurate predictive planning for crop sales and procurement |
| **6. Data Explorer & Export** | Searchable multi-mandi records with instant CSV export | Granular audit trail for researchers and data analysts |

---

## 🚀 Quick Start Guide

### Option 1: Standalone Instant Web Dashboard (Zero Installation)
Double-click `view_standalone_ui.bat` or open `index.html` directly in any modern web browser (Chrome, Edge, Firefox).
- Requires **no Python environment or servers**.
- Fully interactive with Chart.js, Tailwind CSS, live SARIMA simulation, and anomaly detection.

### Option 2: Full Python Streamlit Application
1. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit server:
   ```bash
   python -m streamlit run app.py
   ```
   *(Or double-click `run_dashboard.bat`)*

### Option 3: Jupyter / Google Colab Research Notebook
Open `IntelliMandi_SARIMA_Forecasting.ipynb` in Jupyter Notebook, JupyterLab, or upload to [Google Colab](https://colab.research.google.com/) for a step-by-step reproducible research workflow.

---

## 📁 Repository Structure
```
Intelligent_Mandi_Price_Tracker/
├── app.py                                # Full Streamlit Web Application
├── index.html                            # Zero-dependency client-side interactive dashboard
├── IntelliMandi_SARIMA_Forecasting.ipynb # Complete research & evaluation notebook
├── requirements.txt                      # Python dependencies
├── run_dashboard.bat                     # Windows Streamlit launcher
├── view_standalone_ui.bat                # Windows 1-click browser launcher
├── README.md                             # Architectural & mathematical documentation
├── data/
│   ├── agmarknet_mandi_prices.csv        # Comprehensive daily Agmarknet dataset (22,600+ records)
│   └── data.js                           # Pre-processed JSON payload for index.html
└── src/
    ├── __init__.py
    ├── data_loader.py                    # Agmarknet ingestion, resampling, and unit converter
    ├── trend_analysis.py                 # STL decomposition, ADF test, Bollinger bands, arbitrage
    ├── anomaly_detector.py               # Rolling Z-score, IQR, supply shocks, Isolation Forest
    └── sarima_forecasting.py             # SARIMA model fitting, grid search, validation & diagnostics
```
