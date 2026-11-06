"""
Data Loader & Harmonization Module for Agmarknet Mandi Commodity Prices.
Handles loading, parsing, cleaning, unit conversion, and time series resampling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict

class AgmarknetDataLoader:
    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            # Default to local data directory
            current_dir = Path(__file__).resolve().parent
            data_path = current_dir.parent / "data" / "agmarknet_mandi_prices.csv"
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Loads and pre-processes the Agmarknet dataset."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Agmarknet dataset not found at {self.data_path}")

        # Read CSV
        df = pd.read_csv(self.data_path)

        # Standardize column names if needed
        col_mappings = {
            "Arrival_Date": "Date_Str",
            "Parsed_Date": "Date",
            "Full_Date": "Date",
            "Market": "Mandi",
            "Modal_Price_Rs_Quintal": "Modal_Price",
            "Min_Price_Rs_Quintal": "Min_Price",
            "Max_Price_Rs_Quintal": "Max_Price",
            "Arrival_Tonnes": "Arrival_Tonnes"
        }
        for old_col, new_col in col_mappings.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]

        # Ensure Date format
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        elif "Arrival_Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Arrival_Date"], errors="coerce")

        # Drop invalid dates
        df = df.dropna(subset=["Date"])

        # Ensure numeric columns
        numeric_cols = ["Modal_Price", "Min_Price", "Max_Price", "Arrival_Tonnes"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Fill missing min/max if modal is present
        if "Modal_Price" in df.columns:
            df["Min_Price"] = df["Min_Price"].fillna(df["Modal_Price"] * 0.9)
            df["Max_Price"] = df["Max_Price"].fillna(df["Modal_Price"] * 1.1)

        # Calculate daily spread
        df["Daily_Spread"] = df["Max_Price"] - df["Min_Price"]
        df["Spread_Pct"] = np.where(
            df["Modal_Price"] > 0,
            (df["Daily_Spread"] / df["Modal_Price"]) * 100,
            0
        )

        df = df.sort_values("Date").reset_index(drop=True)
        self.df = df
        return df

    def get_commodities(self) -> List[str]:
        """Returns sorted list of unique commodities."""
        if self.df is None:
            self.load_data()
        return sorted(self.df["Commodity"].dropna().unique().tolist())

    def get_mandis(self, commodity: Optional[str] = None) -> List[str]:
        """Returns unique mandis, optionally filtered by commodity."""
        if self.df is None:
            self.load_data()
        if commodity:
            filtered = self.df[self.df["Commodity"] == commodity]
            return sorted(filtered["Mandi"].dropna().unique().tolist())
        return sorted(self.df["Mandi"].dropna().unique().tolist())

    def get_states(self, commodity: Optional[str] = None) -> List[str]:
        """Returns unique states."""
        if self.df is None:
            self.load_data()
        if commodity:
            filtered = self.df[self.df["Commodity"] == commodity]
            return sorted(filtered["State"].dropna().unique().tolist())
        return sorted(self.df["State"].dropna().unique().tolist())

    def filter_data(
        self,
        commodity: str,
        mandi: Optional[str] = None,
        state: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Filters dataset based on commodity, mandi, state, and date range."""
        if self.df is None:
            self.load_data()

        df_filtered = self.df[self.df["Commodity"] == commodity].copy()

        if mandi and mandi != "All Mandis":
            df_filtered = df_filtered[df_filtered["Mandi"] == mandi]

        if state and state != "All States":
            df_filtered = df_filtered[df_filtered["State"] == state]

        if start_date:
            df_filtered = df_filtered[df_filtered["Date"] >= pd.to_datetime(start_date)]

        if end_date:
            df_filtered = df_filtered[df_filtered["Date"] <= pd.to_datetime(end_date)]

        return df_filtered.sort_values("Date").reset_index(drop=True)

    def get_daily_time_series(
        self,
        commodity: str,
        mandi: Optional[str] = None,
        freq: str = "D",
        interpolate: bool = True
    ) -> pd.Series:
        """
        Creates a clean, continuous daily price time series for SARIMA and trend analysis.
        Handles missing days (Sundays, APMC holidays) using linear interpolation.
        """
        df_filtered = self.filter_data(commodity=commodity, mandi=mandi)
        if df_filtered.empty:
            return pd.Series(dtype=float)

        # If multiple mandis, average the modal price for the commodity
        daily_series = df_filtered.groupby("Date")["Modal_Price"].mean()

        # Reindex to full continuous date range
        full_idx = pd.date_range(start=daily_series.index.min(), end=daily_series.index.max(), freq=freq)
        daily_series = daily_series.reindex(full_idx)

        if interpolate:
            # Linear interpolation for missing dates, then forward/backfill boundaries
            daily_series = daily_series.interpolate(method="linear").bfill().ffill()

        return daily_series

    @staticmethod
    def convert_units(price_val: float, to_unit: str = "quintal") -> float:
        """Converts price between Rs/Quintal (100 kg) and Rs/Kg."""
        if to_unit.lower() == "kg":
            return price_val / 100.0
        return price_val
