# sales_forecaster.py
import os
import warnings
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Try imports that may be optional
try:
    from statsmodels.tsa.arima.model import ARIMA
    _HAS_ARIMA = True
except Exception:
    _HAS_ARIMA = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception:
    _HAS_PROPHET = False

try:
    from xgboost import XGBRegressor
    _HAS_XGBOOST = True
except Exception:
    _HAS_XGBOOST = False

warnings.filterwarnings("ignore")


class SalesForecaster:
    """
    Enhanced SalesForecaster with:
      - Monthly AND Yearly forecasting support
      - Detailed metrics for each forecasting method
      - Per-article model and forecast caching
      - Fast-mode heuristics for small series
    """

    def __init__(self, dataframe: pd.DataFrame,
                 cache_dir: str = "cache",
                 ref_col: str = "Ref Article",
                 date_col: str = "Année",  # Can be year or date
                 sales_col: str = "CA HT NET",
                 frequency: str = "yearly"):  # 'yearly' or 'monthly'
        """
        dataframe: raw dataframe containing at least [ref_col, date_col, sales_col]
        cache_dir: folder to store cached models/forecasts/summaries
        frequency: 'yearly' or 'monthly' for aggregation level
        """
        self.df_raw = dataframe.copy()
        self.ref_col = ref_col
        self.date_col = date_col
        self.sales_col = sales_col
        self.frequency = frequency.lower()

        # ensure standard columns rename when necessary
        self._normalize_column_names()

        self.df_clean = None
        self.grouped_data = None
        self.forecast_results = None

        # cache structure
        self.cache_dir = Path(cache_dir) / self.frequency
        self.model_cache = self.cache_dir / "models"
        self.forecast_cache = self.cache_dir / "forecasts"
        self.summary_cache = self.cache_dir / "summary"
        for p in (self.model_cache, self.forecast_cache, self.summary_cache):
            p.mkdir(parents=True, exist_ok=True)

        # availability flags
        self.has_arima = _HAS_ARIMA
        self.has_prophet = _HAS_PROPHET
        self.has_xgboost = _HAS_XGBOOST

    def _normalize_column_names(self):
        rename_map = {
            'Intitule Marque': 'Marque',
            'Intitule Famille': 'Famille',
            'Intitule Sous Famille': 'Sous Famille'
        }
        for old, new in rename_map.items():
            if old in self.df_raw.columns and new not in self.df_raw.columns:
                self.df_raw.rename(columns={old: new}, inplace=True)

    # -------------------------
    # Data prep with monthly/yearly support
    # -------------------------
    def clean_data(self):
        """Remove rows where sales are null or zero and keep relevant columns"""
        df = self.df_raw
        required = [self.ref_col, self.date_col, self.sales_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in the dataset: {missing}")

        df_clean = df[df[self.sales_col].notna() & (df[self.sales_col] != 0)].copy()
        
        # Handle date column based on frequency
        if self.frequency == 'yearly':
            df_clean[self.date_col] = df_clean[self.date_col].astype(int)
        else:  # monthly
            # Try to parse as datetime if not already
            if not pd.api.types.is_datetime64_any_dtype(df_clean[self.date_col]):
                df_clean[self.date_col] = pd.to_datetime(df_clean[self.date_col], errors='coerce')
            df_clean = df_clean[df_clean[self.date_col].notna()]
            
        self.df_clean = df_clean
        return self.df_clean

    def prepare_data(self):
        """Group by article and period (year or month) summing sales"""
        if self.df_clean is None:
            self.clean_data()

        self.df_clean[self.ref_col] = self.df_clean[self.ref_col].astype(str)

        # Create period column based on frequency
        if self.frequency == 'monthly':
            self.df_clean['period'] = self.df_clean[self.date_col].dt.to_period('M')
        else:  # yearly
            self.df_clean['period'] = self.df_clean[self.date_col]

        available_columns = [c for c in ['Marque', 'Famille', 'Sous Famille', 'Designation'] 
                           if c in self.df_clean.columns]
        agg_dict = {self.sales_col: 'sum'}
        for col in available_columns:
            agg_dict[col] = 'first'

        grouped = self.df_clean.groupby([self.ref_col, 'period']).agg(agg_dict).reset_index()
        grouped[self.ref_col] = grouped[self.ref_col].astype(str)

        self.grouped_data = grouped
        return grouped

    # -------------------------
    # Metric calculation utilities
    # -------------------------
    def calculate_metrics(self, actual, predicted):
        """Calculate comprehensive metrics for a forecast method"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        if len(actual) == 0 or len(predicted) == 0:
            return None
            
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mask = actual != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.nan
            
        # R² score
        try:
            r2 = r2_score(actual, predicted)
        except:
            r2 = np.nan
        
        return {
            'MAE': float(mae),
            'MSE': float(mse),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'R2': float(r2)
        }

    # -------------------------
    # Low-cost forecasting helpers with metrics
    # -------------------------
    def simple_moving_average(self, values, period=3, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        
        if len(values) < period:
            forecast = float(np.mean(values))
        else:
            forecast = float(np.mean(values[-period:]))
        
        if return_metrics and len(values) > period:
            # Calculate metrics on historical predictions
            predictions = []
            actuals = []
            for i in range(period, len(values)):
                pred = np.mean(values[i-period:i])
                predictions.append(pred)
                actuals.append(values[i])
            metrics = self.calculate_metrics(actuals, predictions)
        else:
            metrics = None
            
        return (forecast, metrics) if return_metrics else forecast

    def exponential_smoothing(self, values, alpha=0.3, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        if len(values) == 1:
            return (float(values[0]), None) if return_metrics else float(values[0])
        
        # Calculate forecast and predictions for metrics
        predictions = []
        f = values[0]
        for i, v in enumerate(values[1:], 1):
            predictions.append(f)
            f = alpha * v + (1 - alpha) * f
        
        forecast = alpha * values[-1] + (1 - alpha) * f
        
        if return_metrics and len(predictions) > 0:
            metrics = self.calculate_metrics(values[1:], predictions)
        else:
            metrics = None
            
        return (float(forecast), metrics) if return_metrics else float(forecast)

    def linear_regression_forecast(self, periods, values, return_metrics=False):
        if len(periods) == 0:
            return (0.0, None) if return_metrics else 0.0
        
        # Convert periods to numeric
        if self.frequency == 'monthly':
            X = np.array([p.ordinal for p in periods]).reshape(-1, 1)
            next_period = periods[-1] + 1
            next_X = np.array([[next_period.ordinal]])
        else:
            X = np.array(periods).reshape(-1, 1)
            next_X = np.array([[int(max(periods) + 1)]])
        
        y = np.array(values)
        model = LinearRegression()
        model.fit(X, y)
        
        forecast = float(model.predict(next_X)[0])
        forecast = max(0.0, forecast)
        
        if return_metrics:
            predictions = model.predict(X)
            metrics = self.calculate_metrics(y, predictions)
        else:
            metrics = None
            
        return (forecast, metrics) if return_metrics else forecast

    # -------------------------
    # Time series / ML forecasts with metrics
    # -------------------------
    def arima_forecast(self, values, order=(1, 1, 0), return_metrics=False):
        if len(values) == 0 or not self.has_arima:
            return (0.0, None) if return_metrics else 0.0
        try:
            model = ARIMA(values, order=order)
            fit = model.fit()
            forecast = float(fit.forecast(steps=1)[0])
            forecast = max(0.0, forecast)
            
            if return_metrics and len(values) > 3:
                # In-sample predictions
                predictions = fit.fittedvalues
                if len(predictions) == len(values):
                    metrics = self.calculate_metrics(values, predictions)
                else:
                    metrics = None
            else:
                metrics = None
                
            return (forecast, metrics) if return_metrics else forecast
        except Exception:
            return (0.0, None) if return_metrics else 0.0

    def prophet_forecast(self, periods, values, return_metrics=False):
        if len(values) < 2 or not self.has_prophet:
            return (0.0, None) if return_metrics else 0.0
        try:
            # Convert periods to datetime
            if self.frequency == 'monthly':
                dates = [p.to_timestamp() for p in periods]
            else:
                dates = pd.to_datetime(periods, format='%Y')
            
            dfp = pd.DataFrame({'ds': dates, 'y': values})
            m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            m.fit(dfp)
            
            freq = 'M' if self.frequency == 'monthly' else 'Y'
            future = m.make_future_dataframe(periods=1, freq=freq)
            fc = m.predict(future)
            forecast = float(max(0.0, fc.iloc[-1]['yhat']))
            
            if return_metrics:
                in_sample = m.predict(dfp)
                metrics = self.calculate_metrics(values, in_sample['yhat'].values)
            else:
                metrics = None
                
            return (forecast, metrics) if return_metrics else forecast
        except Exception as e:
            return (0.0, None) if return_metrics else 0.0

    def xgboost_forecast(self, periods, values, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        
        # Convert periods to numeric
        if self.frequency == 'monthly':
            X = np.array([p.ordinal for p in periods]).reshape(-1, 1)
            next_period = periods[-1] + 1
            next_X = np.array([[next_period.ordinal]])
        else:
            X = np.array(periods).reshape(-1, 1)
            next_X = np.array([[int(max(periods) + 1)]])
        
        y = np.array(values)
        
        try:
            if self.has_xgboost:
                model = XGBRegressor(n_estimators=200, learning_rate=0.05, verbosity=0, n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            forecast = float(model.predict(next_X)[0])
            forecast = max(0.0, forecast)
            
            if return_metrics:
                predictions = model.predict(X)
                metrics = self.calculate_metrics(y, predictions)
            else:
                metrics = None
                
            return (forecast, metrics) if return_metrics else forecast
        except Exception:
            return (0.0, None) if return_metrics else 0.0

    # -------------------------
    # Caching utilities
    # -------------------------
    def _model_cache_path(self, ref_article, model_name):
        safe = str(ref_article).replace("/", "_").replace(" ", "_")
        return self.model_cache / f"{safe}__{model_name}.pkl"

    def _forecast_cache_path(self, ref_article):
        safe = str(ref_article).replace("/", "_").replace(" ", "_")
        return self.forecast_cache / f"{safe}__forecast.csv"

    def _summary_cache_path(self):
        return self.summary_cache / "summary.parquet"

    # -------------------------
    # Per-article orchestration
    # -------------------------
    def _get_article_series(self, ref_article):
        """Return sorted (periods, values) arrays for the article"""
        if self.grouped_data is None:
            self.prepare_data()
        df = self.grouped_data[self.grouped_data[self.ref_col] == ref_article].sort_values('period')
        if df.empty:
            return [], [], {}
        
        periods = df['period'].tolist()
        values = df[self.sales_col].astype(float).tolist()
        
        # metadata
        metadata = {}
        for c in ['Designation', 'Marque', 'Famille']:
            if c in df.columns:
                metadata[c.lower()] = df[c].iloc[0]
            else:
                metadata[c.lower()] = None
        return periods, values, metadata

    def _load_forecast_cache(self, ref_article, use_cache=True):
        p = self._forecast_cache_path(ref_article)
        if use_cache and p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
        return None

    def _save_forecast_cache(self, ref_article, df):
        p = self._forecast_cache_path(ref_article)
        df.to_csv(p, index=False)

    # -------------------------
    # Main per-article forecast method with detailed metrics
    # -------------------------
    def forecast_article(self, ref_article, period=3, alpha=0.3,
                         force_recompute=False, include_methods=None,
                         fast_mode=True, return_metrics=True):
        """
        Forecast for a single article with detailed metrics for each method.
        """
        import json
        
        if include_methods is None:
            include_methods = ['SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST']

        cached = None if force_recompute else self._load_forecast_cache(ref_article, use_cache=True)
        if cached is not None:
            return cached.to_dict(orient='records')[0] if not cached.empty else None

        periods, values, meta = self._get_article_series(ref_article)
        if len(values) == 0:
            return None

        # Determine next period
        if self.frequency == 'monthly':
            next_period = str(periods[-1] + 1)
        else:
            next_period = int(max(periods) + 1)

        # Fast-mode heuristics
        if fast_mode and len(values) < 6:
            allowed = [m for m in include_methods if m in ['SMA', 'ExpSmoothing', 'LinearReg', 'XGBOOST']]
            include_methods = allowed

        # Compute forecasts and metrics
        results = {}
        metrics_dict = {}
        
        if 'SMA' in include_methods:
            forecast, metrics = self.simple_moving_average(values, period, return_metrics=True)
            results['sma_forecast'] = forecast
            metrics_dict['sma_metrics'] = metrics
        
        if 'ExpSmoothing' in include_methods:
            forecast, metrics = self.exponential_smoothing(values, alpha, return_metrics=True)
            results['es_forecast'] = forecast
            metrics_dict['es_metrics'] = metrics
        
        if 'LinearReg' in include_methods:
            forecast, metrics = self.linear_regression_forecast(periods, values, return_metrics=True)
            results['lr_forecast'] = forecast
            metrics_dict['lr_metrics'] = metrics
        
        if 'ARIMA' in include_methods and self.has_arima and len(values) >= 3:
            forecast, metrics = self.arima_forecast(values, return_metrics=True)
            results['arima_forecast'] = forecast
            metrics_dict['arima_metrics'] = metrics
        
        if 'PROPHET' in include_methods and self.has_prophet and len(values) >= 3:
            forecast, metrics = self.prophet_forecast(periods, values, return_metrics=True)
            results['prophet_forecast'] = forecast
            metrics_dict['prophet_metrics'] = metrics
        
        if 'XGBOOST' in include_methods:
            forecast, metrics = self.xgboost_forecast(periods, values, return_metrics=True)
            results['xgb_forecast'] = forecast
            metrics_dict['xgb_metrics'] = metrics

        # Build ensemble average
        method_values = [v for v in results.values() if v is not None and not np.isnan(v)]
        avg_forecast = float(np.mean(method_values)) if len(method_values) > 0 else 0.0

        # Stats
        avg_sales = float(np.mean(values))
        max_sales = float(np.max(values))
        min_sales = float(np.min(values))
        std_sales = float(np.std(values))
        trend_pct = float(((values[-1] - values[0]) / values[0]) * 100) if values[0] != 0 else 0.0

        result = {
            'ref_article': ref_article,
            'designation': meta.get('designation'),
            'marque': meta.get('marque'),
            'famille': meta.get('famille'),
            'frequency': self.frequency,
            'next_period': next_period,
            'avg_forecast': float(avg_forecast),
            # FIXED: Convert to lists and JSON serialize for proper CSV storage/loading
            'historical_periods': json.dumps([str(p) for p in periods]),
            'historical_values': json.dumps([float(v) for v in values]),
            'avg_sales': avg_sales,
            'max_sales': max_sales,
            'min_sales': min_sales,
            'std_sales': std_sales,
            'trend_pct': trend_pct,
            'data_points': len(values)
        }
        
        # Add individual forecasts
        for key, val in results.items():
            result[key] = float(val) if val is not None else np.nan
        
        # Add metrics as JSON strings (for CSV storage)
        for key, metrics in metrics_dict.items():
            if metrics:
                result[key] = json.dumps(metrics)
            else:
                result[key] = None

        # Save cache
        df_out = pd.DataFrame([result])
        self._save_forecast_cache(ref_article, df_out)

        return result

    # -------------------------
    # Forecast all articles
    # -------------------------
    def forecast_all_articles(self, period=3, alpha=0.3, force_recompute=False,
                              include_methods=None, fast_mode=True, progress_callback=None):
        if self.grouped_data is None:
            self.prepare_data()

        articles = sorted(self.grouped_data[self.ref_col].unique().tolist())
        all_results = []
        total = len(articles)
        for i, a in enumerate(articles, start=1):
            if progress_callback:
                try:
                    progress_callback(i, total)
                except Exception:
                    pass
            res = self.forecast_article(a, period=period, alpha=alpha, force_recompute=force_recompute,
                                        include_methods=include_methods, fast_mode=fast_mode)
            if res:
                all_results.append(res)

        df_all = pd.DataFrame(all_results)
        if not df_all.empty:
            df_all['ref_article'] = df_all['ref_article'].astype(str)
            df_all = df_all.sort_values('avg_forecast', ascending=False).reset_index(drop=True)

        summary_path = self._summary_cache_path()
        df_all.to_parquet(summary_path, index=False)

        self.forecast_results = df_all
        return df_all

    # -------------------------
    # Utility methods
    # -------------------------
    def classify_trend_label(self, last_actual_mean, next_forecast_mean, tol=0.05):
        if last_actual_mean == 0:
            return "Stable"
        change = (next_forecast_mean - last_actual_mean) / last_actual_mean
        if change > tol:
            return "Uptrend"
        elif change < -tol:
            return "Downtrend"
        else:
            return "Stable"

    def generate_summary(self, lookback_periods=3, force_recompute=False):
        summary_path = self._summary_cache_path()
        if (not force_recompute) and summary_path.exists():
            try:
                df = pd.read_parquet(summary_path)
                self.forecast_results = df
                return df
            except Exception:
                pass

        if self.grouped_data is None:
            self.prepare_data()

        articles = sorted(self.grouped_data[self.ref_col].unique().tolist())
        recs = []
        for a in articles:
            f = self._load_forecast_cache(a, use_cache=True)
            if f is None:
                res = self.forecast_article(a, fast_mode=True)
                f = pd.DataFrame([res]) if res is not None else None
            if f is None or f.empty:
                continue

            row = f.iloc[0].to_dict()
            vals = row.get('historical_values', [])
            if len(vals) == 0:
                continue

            if lookback_periods <= 0:
                last_mean = float(np.mean(vals))
            else:
                last_vals = vals[-lookback_periods:] if len(vals) >= lookback_periods else vals
                last_mean = float(np.mean(last_vals)) if last_vals else 0.0

            next_forecast = float(row.get('avg_forecast', 0.0))
            trend_label = self.classify_trend_label(last_mean, next_forecast, tol=0.05)

            recs.append({
                'ref_article': row['ref_article'],
                'designation': row.get('designation'),
                'marque': row.get('marque'),
                'famille': row.get('famille'),
                'frequency': row.get('frequency'),
                'next_period': row.get('next_period'),
                'avg_forecast': float(row.get('avg_forecast', 0.0)),
                'trend_pct': float(row.get('trend_pct', 0.0)),
                'trend_label': trend_label,
                'data_points': int(row.get('data_points', 0))
            })

        df_summary = pd.DataFrame(recs)
        if not df_summary.empty:
            df_summary['ref_article'] = df_summary['ref_article'].astype(str)
            df_summary = df_summary.sort_values('avg_forecast', ascending=False).reset_index(drop=True)
            df_summary.to_parquet(summary_path, index=False)
        self.forecast_results = df_summary
        return df_summary

    def clear_model_cache(self):
        for p in self.model_cache.glob("*.pkl"):
            try:
                p.unlink()
            except Exception:
                pass

    def clear_forecast_cache(self):
        for p in self.forecast_cache.glob("*.csv"):
            try:
                p.unlink()
            except Exception:
                pass

    def clear_summary_cache(self):
        sp = self._summary_cache_path()
        if sp.exists():
            try:
                sp.unlink()
            except Exception:
                pass

    def clear_all_caches(self):
        self.clear_model_cache()
        self.clear_forecast_cache()
        self.clear_summary_cache()