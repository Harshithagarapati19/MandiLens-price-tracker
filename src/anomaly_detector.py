"""
Comprehensive Anomaly Detection Engine for Mandi Commodity Prices.
Combines Dynamic Rolling Z-Score, Interquartile Range (IQR),
Arrival Supply-Shock Anomaly Detection, and Unsupervised Isolation Forest.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from sklearn.ensemble import IsolationForest

class MandiAnomalyDetector:
    def __init__(self, df: pd.DataFrame):
        """
        Expects a DataFrame containing 'Date', 'Modal_Price', and optionally
        'Arrival_Tonnes', 'Min_Price', 'Max_Price'.
        """
        self.df = df.copy()
        if "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df = self.df.sort_values("Date").reset_index(drop=True)

    def detect_zscore_anomalies(
        self,
        window: int = 14,
        threshold: float = 2.5
    ) -> pd.DataFrame:
        """
        Rolling Z-Score Anomaly Detector:
        Z_t = (P_t - rolling_mean) / rolling_std
        Flags observations where |Z_t| >= threshold.
        """
        res = self.df.copy()
        rolling_mean = res["Modal_Price"].rolling(window=window, min_periods=3).mean()
        rolling_std = res["Modal_Price"].rolling(window=window, min_periods=3).std().replace(0, np.nan)

        res["rolling_mean"] = rolling_mean
        res["rolling_std"] = rolling_std
        res["z_score"] = (res["Modal_Price"] - rolling_mean) / rolling_std
        res["z_score"] = res["z_score"].fillna(0)

        res["is_zscore_anomaly"] = res["z_score"].abs() >= threshold
        res["zscore_anomaly_type"] = np.where(
            res["z_score"] >= threshold,
            "Price Spike (Surge)",
            np.where(res["z_score"] <= -threshold, "Price Crash (Plunge)", "Normal")
        )
        return res

    def detect_iqr_anomalies(self) -> pd.DataFrame:
        """
        Interquartile Range (IQR) Non-Parametric Anomaly Detector:
        Lower Bound = Q1 - 1.5 * IQR
        Upper Bound = Q3 + 1.5 * IQR
        """
        res = self.df.copy()
        q1 = res["Modal_Price"].quantile(0.25)
        q3 = res["Modal_Price"].quantile(0.75)
        iqr = q3 - q1
        lower_bound = max(0, q1 - (1.5 * iqr))
        upper_bound = q3 + (1.5 * iqr)

        res["iqr_lower_bound"] = lower_bound
        res["iqr_upper_bound"] = upper_bound
        res["is_iqr_anomaly"] = (res["Modal_Price"] < lower_bound) | (res["Modal_Price"] > upper_bound)
        return res

    def detect_supply_shock_anomalies(self) -> pd.DataFrame:
        """
        Multivariate Supply Shock Detector:
        Examines joint shifts in Arrival Volumes (Tonnes) and Modal Price.
        - Supply Deficit Shock: Arrivals drop < 20th percentile AND Price surges > 80th percentile.
        - Market Glut Crash: Arrivals surge > 80th percentile AND Price plunges < 20th percentile.
        """
        res = self.df.copy()
        if "Arrival_Tonnes" not in res.columns or res["Arrival_Tonnes"].dropna().empty:
            res["is_supply_shock"] = False
            res["supply_shock_type"] = "Normal"
            return res

        price_p20 = res["Modal_Price"].quantile(0.20)
        price_p80 = res["Modal_Price"].quantile(0.80)
        arrival_p20 = res["Arrival_Tonnes"].quantile(0.20)
        arrival_p80 = res["Arrival_Tonnes"].quantile(0.80)

        is_deficit_shock = (res["Arrival_Tonnes"] <= arrival_p20) & (res["Modal_Price"] >= price_p80)
        is_glut_crash = (res["Arrival_Tonnes"] >= arrival_p80) & (res["Modal_Price"] <= price_p20)

        res["is_supply_shock"] = is_deficit_shock | is_glut_crash
        res["supply_shock_type"] = np.where(
            is_deficit_shock,
            "Supply Deficit Shock (Low Arrival + High Price)",
            np.where(is_glut_crash, "Market Glut Crash (High Arrival + Low Price)", "Normal")
        )
        return res

    def detect_isolation_forest_anomalies(
        self,
        contamination: float = 0.04,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Unsupervised Machine Learning Anomaly Detection via Isolation Forest.
        Features engineered:
        1. 1-day percentage price change
        2. 7-day rolling price standard deviation
        3. Intra-day price spread ratio (Max - Min) / Modal
        4. (Optional) Arrival volume percentage change
        """
        res = self.df.copy()
        if len(res) < 15:
            res["is_iforest_anomaly"] = False
            res["iforest_score"] = 0.0
            return res

        # Feature engineering
        features = pd.DataFrame()
        features["price_return"] = res["Modal_Price"].pct_change().fillna(0)
        features["price_std_7"] = res["Modal_Price"].rolling(7, min_periods=1).std().fillna(0)
        
        if "Max_Price" in res.columns and "Min_Price" in res.columns:
            features["spread_ratio"] = np.where(
                res["Modal_Price"] > 0,
                (res["Max_Price"] - res["Min_Price"]) / res["Modal_Price"],
                0
            )
        else:
            features["spread_ratio"] = 0

        if "Arrival_Tonnes" in res.columns:
            features["arrival_return"] = res["Arrival_Tonnes"].pct_change().fillna(0)

        # Replace infinite values
        features = features.replace([np.inf, -np.inf], 0).fillna(0)

        iso = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        preds = iso.fit_predict(features)
        scores = iso.decision_function(features)

        res["is_iforest_anomaly"] = (preds == -1)
        res["iforest_score"] = scores
        return res

    def run_unified_anomaly_detection(
        self,
        z_threshold: float = 2.5,
        contamination: float = 0.04
    ) -> pd.DataFrame:
        """
        Executes all anomaly detection engines and unifies results with severity scoring.
        Severity classification:
        - Critical: Flagged by 2+ detectors or Extreme Z-score >= 3.5
        - Moderate: Flagged by 1 detector with Z-score >= 2.5
        - Low: Mild deviation
        """
        z_df = self.detect_zscore_anomalies(threshold=z_threshold)
        iqr_df = self.detect_iqr_anomalies()
        shock_df = self.detect_supply_shock_anomalies()
        iso_df = self.detect_isolation_forest_anomalies(contamination=contamination)

        unified = self.df.copy()
        unified["z_score"] = z_df["z_score"]
        unified["rolling_mean"] = z_df["rolling_mean"]
        unified["rolling_std"] = z_df["rolling_std"]
        unified["is_z_anomaly"] = z_df["is_zscore_anomaly"]
        unified["z_anomaly_type"] = z_df["zscore_anomaly_type"]

        unified["is_iqr_anomaly"] = iqr_df["is_iqr_anomaly"]
        unified["is_supply_shock"] = shock_df["is_supply_shock"]
        unified["supply_shock_type"] = shock_df["supply_shock_type"]

        unified["is_iforest_anomaly"] = iso_df["is_iforest_anomaly"]
        unified["iforest_score"] = iso_df["iforest_score"]

        # Anomaly Vote count (0 to 4)
        unified["anomaly_votes"] = (
            unified["is_z_anomaly"].astype(int) +
            unified["is_iqr_anomaly"].astype(int) +
            unified["is_supply_shock"].astype(int) +
            unified["is_iforest_anomaly"].astype(int)
        )

        unified["is_anomaly"] = unified["anomaly_votes"] >= 1

        # Severity categorization
        def assign_severity(row):
            if not row["is_anomaly"]:
                return "Normal"
            if row["anomaly_votes"] >= 2 or abs(row["z_score"]) >= 3.5:
                return "Critical Price Shock"
            elif row["anomaly_votes"] == 1:
                return "Moderate Deviation"
            return "Low Warning"

        unified["severity"] = unified.apply(assign_severity, axis=1)

        # Primary Cause explanation
        def explain_cause(row):
            if not row["is_anomaly"]:
                return "Normal Trading Conditions"
            causes = []
            if row["is_supply_shock"]:
                causes.append(row["supply_shock_type"])
            if row["is_z_anomaly"]:
                causes.append(f"Z-Score spike ({row['z_score']:.1f}σ)")
            if row["is_iqr_anomaly"]:
                causes.append("IQR Boundary Exceeded")
            if row["is_iforest_anomaly"]:
                causes.append("Isolation Forest Outlier")
            return " | ".join(causes) if causes else "Statistical Outlier"

        unified["anomaly_reason"] = unified.apply(explain_cause, axis=1)

        return unified
