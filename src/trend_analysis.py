"""
Trend and Seasonality Analysis Engine for Agmarknet Mandi Commodity Prices.
Includes STL / Seasonal Decomposition, Augmented Dickey-Fuller (ADF) Stationarity Testing,
Moving Averages, Bollinger Bands, and Inter-Mandi Arbitrage Spread Analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

class MandiTrendAnalyzer:
    def __init__(self, time_series: pd.Series):
        """
        Initializes the analyzer with a continuous pandas Series indexed by DatetimeIndex.
        """
        self.series = time_series.dropna().sort_index()

    def check_stationarity_adf(self) -> Dict[str, Any]:
        """
        Performs Augmented Dickey-Fuller (ADF) Test to determine time series stationarity.
        Null Hypothesis (H0): The series has a unit root (is non-stationary).
        Alternative Hypothesis (H1): The series is stationary.
        """
        if len(self.series) < 15:
            return {
                "adf_statistic": 0.0,
                "p_value": 1.0,
                "is_stationary": False,
                "critical_values": {},
                "recommendation": "Insufficient data points for ADF test (min 15 required)."
            }

        try:
            result = adfuller(self.series.values, autolag="AIC")
            adf_stat = float(result[0])
            p_val = float(result[1])
            crit_vals = {k: float(v) for k, v in result[4].items()}
            is_stat = p_val < 0.05

            recommendation = (
                "Series is stationary (p < 0.05). Differencing order d=0 is suitable for SARIMA."
                if is_stat
                else "Series is non-stationary (p >= 0.05). First-order differencing d=1 is recommended for SARIMA."
            )

            return {
                "adf_statistic": adf_stat,
                "p_value": p_val,
                "is_stationary": is_stat,
                "critical_values": crit_vals,
                "used_lag": int(result[2]),
                "n_obs": int(result[3]),
                "recommendation": recommendation
            }
        except Exception as e:
            return {
                "adf_statistic": 0.0,
                "p_value": 1.0,
                "is_stationary": False,
                "critical_values": {},
                "recommendation": f"ADF test failed: {str(e)}"
            }

    def decompose_time_series(
        self,
        period: int = 7,
        model: str = "additive"
    ) -> Optional[Dict[str, pd.Series]]:
        """
        Decomposes the time series into Trend, Seasonal, and Residual components.
        period: 7 for weekly agricultural cycle, 30 for monthly.
        """
        if len(self.series) < 2 * period:
            return None

        try:
            # Handle strictly positive requirement for multiplicative model
            if model == "multiplicative" and (self.series <= 0).any():
                model = "additive"

            decomp = seasonal_decompose(
                self.series,
                model=model,
                period=period,
                extrapolate_trend="freq"
            )

            return {
                "observed": self.series,
                "trend": decomp.trend,
                "seasonal": decomp.seasonal,
                "residual": decomp.resid,
                "model": model,
                "period": period
            }
        except Exception:
            # Fallback to additive if anything fails
            try:
                decomp = seasonal_decompose(self.series, model="additive", period=period, extrapolate_trend="freq")
                return {
                    "observed": self.series,
                    "trend": decomp.trend,
                    "seasonal": decomp.seasonal,
                    "residual": decomp.resid,
                    "model": "additive",
                    "period": period
                }
            except Exception:
                return None

    def calculate_technical_indicators(self) -> pd.DataFrame:
        """
        Computes Moving Averages (SMA 7, 14, 30), EMA 14, Bollinger Bands,
        and rolling volatility for trend identification.
        """
        df = pd.DataFrame({"price": self.series})
        df["sma_7"] = df["price"].rolling(window=7, min_periods=1).mean()
        df["sma_14"] = df["price"].rolling(window=14, min_periods=1).mean()
        df["sma_30"] = df["price"].rolling(window=30, min_periods=1).mean()
        df["ema_14"] = df["price"].ewm(span=14, adjust=False).mean()

        # Bollinger Bands (20-day window, 2 std deviations)
        rolling_20_mean = df["price"].rolling(window=20, min_periods=3).mean()
        rolling_20_std = df["price"].rolling(window=20, min_periods=3).std().fillna(0)
        df["bollinger_mid"] = rolling_20_mean
        df["bollinger_upper"] = rolling_20_mean + (2 * rolling_20_std)
        df["bollinger_lower"] = np.maximum(rolling_20_mean - (2 * rolling_20_std), 0)

        # 7-day Momentum (% change)
        df["momentum_7d_pct"] = df["price"].pct_change(periods=7) * 100

        # Rolling Volatility (Coefficient of Variation)
        df["rolling_cv_pct"] = np.where(
            rolling_20_mean > 0,
            (rolling_20_std / rolling_20_mean) * 100,
            0
        )

        return df

    @staticmethod
    def analyze_cross_mandi_arbitrage(
        df_filtered: pd.DataFrame,
        target_date: Optional[pd.Timestamp] = None
    ) -> Dict[str, Any]:
        """
        Analyzes spatial price spreads across mandis for a given commodity.
        Identifies highest-paying mandi (selling hub) vs lowest-paying mandi (procurement hub)
        and computes the inter-mandi gross arbitrage margin.
        """
        if df_filtered.empty:
            return {}

        if target_date is None:
            target_date = df_filtered["Date"].max()

        latest_day = df_filtered[df_filtered["Date"] == target_date]
        if latest_day.empty:
            latest_day = df_filtered[df_filtered["Date"] == df_filtered["Date"].max()]

        if latest_day.empty:
            return {}

        # Mandi price stats for the day
        mandi_summary = latest_day.groupby(["Mandi", "State"], as_index=False)["Modal_Price"].mean()
        if len(mandi_summary) < 2:
            return {
                "date": target_date,
                "mandi_count": len(mandi_summary),
                "has_arbitrage": False,
                "summary": mandi_summary
            }

        highest_mandi = mandi_summary.loc[mandi_summary["Modal_Price"].idxmax()]
        lowest_mandi = mandi_summary.loc[mandi_summary["Modal_Price"].idxmin()]

        spread_diff = float(highest_mandi["Modal_Price"] - lowest_mandi["Modal_Price"])
        spread_pct = float((spread_diff / lowest_mandi["Modal_Price"]) * 100) if lowest_mandi["Modal_Price"] > 0 else 0.0

        return {
            "date": target_date,
            "mandi_count": len(mandi_summary),
            "has_arbitrage": True,
            "highest_mandi": highest_mandi["Mandi"],
            "highest_state": highest_mandi["State"],
            "highest_price": float(highest_mandi["Modal_Price"]),
            "lowest_mandi": lowest_mandi["Mandi"],
            "lowest_state": lowest_mandi["State"],
            "lowest_price": float(lowest_mandi["Modal_Price"]),
            "gross_spread_rs": spread_diff,
            "spread_percentage": spread_pct,
            "mandi_summary": mandi_summary.sort_values("Modal_Price", ascending=False).to_dict(orient="records")
        }
