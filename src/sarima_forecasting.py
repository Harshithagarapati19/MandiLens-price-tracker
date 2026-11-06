"""
SARIMA (Seasonal AutoRegressive Integrated Moving Average) Forecasting Engine.
Provides model fitting, auto-order search, train-test evaluation, multi-horizon out-of-sample
forecasting with confidence intervals, and residual diagnostics.
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress convergence warnings during grid search
warnings.filterwarnings("ignore")

class MandiSARIMAForecaster:
    def __init__(self, time_series: pd.Series, seasonal_period: int = 7):
        """
        time_series: pd.Series with DatetimeIndex and numeric price values.
        seasonal_period: default 7 (weekly mandi trading cycle).
        """
        self.series = time_series.dropna().sort_index()
        self.seasonal_period = seasonal_period
        self.model_fit: Optional[SARIMAXResults] = None
        self.best_order: Tuple[int, int, int] = (1, 1, 1)
        self.best_seasonal_order: Tuple[int, int, int, int] = (1, 0, 1, seasonal_period)

    def evaluate_train_test(
        self,
        order: Tuple[int, int, int] = (1, 1, 1),
        seasonal_order: Optional[Tuple[int, int, int, int]] = None,
        test_size: int = 28
    ) -> Dict[str, Any]:
        """
        Splits series into train and test sets, fits SARIMA, and calculates validation metrics:
        MAE, RMSE, MAPE, and Directional Accuracy.
        """
        if seasonal_order is None:
            seasonal_order = (1, 0, 1, self.seasonal_period)

        n = len(self.series)
        if n < test_size + 14:
            test_size = max(7, int(n * 0.15))

        train = self.series.iloc[:-test_size]
        test = self.series.iloc[-test_size:]

        try:
            model = SARIMAX(
                train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fit_res = model.fit(disp=False, maxiter=100)

            # Predict on test horizon
            pred_res = fit_res.get_forecast(steps=test_size)
            predictions = pred_res.predicted_mean
            ci = pred_res.conf_int(alpha=0.05)

            actual = test.values
            pred_vals = predictions.values

            mae = float(mean_absolute_error(actual, pred_vals))
            rmse = float(np.sqrt(mean_squared_error(actual, pred_vals)))
            mape = float(np.mean(np.abs((actual - pred_vals) / np.maximum(actual, 1e-5))) * 100)

            # Directional accuracy
            if len(actual) > 1:
                actual_dir = np.sign(np.diff(actual))
                pred_dir = np.sign(np.diff(pred_vals))
                directional_acc = float(np.mean(actual_dir == pred_dir) * 100)
            else:
                directional_acc = 100.0

            return {
                "order": order,
                "seasonal_order": seasonal_order,
                "train_size": len(train),
                "test_size": len(test),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
                "directional_accuracy_pct": round(directional_acc, 2),
                "aic": round(fit_res.aic, 2),
                "bic": round(fit_res.bic, 2),
                "test_index": test.index,
                "actual": actual,
                "predictions": pred_vals,
                "conf_int_lower": ci.iloc[:, 0].values,
                "conf_int_upper": ci.iloc[:, 1].values,
                "success": True
            }

        except Exception as e:
            # Fallback naive forecast
            last_val = train.iloc[-1]
            actual = test.values
            pred_vals = np.full(len(actual), last_val)
            mae = float(mean_absolute_error(actual, pred_vals))
            rmse = float(np.sqrt(mean_squared_error(actual, pred_vals)))
            mape = float(np.mean(np.abs((actual - pred_vals) / np.maximum(actual, 1e-5))) * 100)

            return {
                "order": order,
                "seasonal_order": seasonal_order,
                "train_size": len(train),
                "test_size": len(test),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "mape": round(mape, 2),
                "directional_accuracy_pct": 50.0,
                "aic": 99999.0,
                "bic": 99999.0,
                "test_index": test.index,
                "actual": actual,
                "predictions": pred_vals,
                "conf_int_lower": pred_vals * 0.9,
                "conf_int_upper": pred_vals * 1.1,
                "success": False,
                "error": str(e)
            }

    def auto_fit_best_order(
        self,
        p_range: List[int] = [0, 1, 2],
        d_range: List[int] = [0, 1],
        q_range: List[int] = [0, 1, 2],
        P_range: List[int] = [0, 1],
        D_range: List[int] = [0, 1],
        Q_range: List[int] = [0, 1]
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]:
        """
        Lightweight Grid-Search over candidate SARIMA parameter configurations,
        selecting the order that minimizes Akaike Information Criterion (AIC).
        """
        best_aic = float("inf")
        best_cfg = ((1, 1, 1), (1, 0, 1, self.seasonal_period))

        # Sample lightweight candidates for fast interactive responsiveness
        candidate_orders = [
            ((1, 1, 1), (1, 0, 1, self.seasonal_period)),
            ((1, 1, 2), (0, 1, 1, self.seasonal_period)),
            ((2, 1, 1), (1, 0, 0, self.seasonal_period)),
            ((1, 0, 1), (1, 0, 1, self.seasonal_period)),
            ((0, 1, 1), (0, 1, 1, self.seasonal_period)),
            ((2, 1, 2), (1, 1, 1, self.seasonal_period))
        ]

        for order, s_order in candidate_orders:
            try:
                model = SARIMAX(
                    self.series,
                    order=order,
                    seasonal_order=s_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
                res = model.fit(disp=False, maxiter=50)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_cfg = (order, s_order)
            except Exception:
                continue

        self.best_order, self.best_seasonal_order = best_cfg
        return best_cfg

    def forecast_future(
        self,
        steps: int = 30,
        order: Optional[Tuple[int, int, int]] = None,
        seasonal_order: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Fits SARIMA model on the full historical series and projects forward
        for 'steps' days with 80% and 95% confidence intervals.
        """
        if order is None:
            order = self.best_order
        if seasonal_order is None:
            seasonal_order = self.best_seasonal_order

        try:
            model = SARIMAX(
                self.series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            self.model_fit = model.fit(disp=False, maxiter=100)

            # Generate future date index
            last_date = self.series.index.max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="D")

            # Get forecast with 95% CI
            forecast_95 = self.model_fit.get_forecast(steps=steps)
            pred_mean = forecast_95.predicted_mean
            ci_95 = forecast_95.conf_int(alpha=0.05)

            # 80% CI
            ci_80 = forecast_95.conf_int(alpha=0.20)

            # Residuals for diagnostics
            residuals = self.model_fit.resid

            # Diagnostic summary
            diag = {
                "aic": float(self.model_fit.aic),
                "bic": float(self.model_fit.bic),
                "log_likelihood": float(self.model_fit.llf),
                "residual_mean": float(residuals.mean()),
                "residual_std": float(residuals.std())
            }

            return {
                "future_dates": future_dates,
                "forecast_mean": pred_mean.values,
                "ci_95_lower": np.maximum(ci_95.iloc[:, 0].values, 0),
                "ci_95_upper": ci_95.iloc[:, 1].values,
                "ci_80_lower": np.maximum(ci_80.iloc[:, 0].values, 0),
                "ci_80_upper": ci_80.iloc[:, 1].values,
                "order": order,
                "seasonal_order": seasonal_order,
                "diagnostics": diag,
                "residuals": residuals,
                "success": True
            }

        except Exception as e:
            # Fallback projection based on recent trend and seasonal average
            last_date = self.series.index.max()
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq="D")
            recent_mean = float(self.series.tail(14).mean())
            recent_std = float(self.series.tail(14).std())

            preds = np.full(steps, recent_mean)
            ci_95_l = np.maximum(preds - (1.96 * recent_std), 0)
            ci_95_u = preds + (1.96 * recent_std)
            ci_80_l = np.maximum(preds - (1.28 * recent_std), 0)
            ci_80_u = preds + (1.28 * recent_std)

            return {
                "future_dates": future_dates,
                "forecast_mean": preds,
                "ci_95_lower": ci_95_l,
                "ci_95_upper": ci_95_u,
                "ci_80_lower": ci_80_l,
                "ci_80_upper": ci_80_u,
                "order": order,
                "seasonal_order": seasonal_order,
                "diagnostics": {"aic": 0, "bic": 0, "log_likelihood": 0, "residual_mean": 0, "residual_std": 0},
                "residuals": pd.Series(dtype=float),
                "success": False,
                "error": str(e)
            }
