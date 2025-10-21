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


class IntegratedForecaster:
    """
    Enhanced IntegratedForecaster for simultaneous Sales AND Quantity forecasting:
      - Monthly AND Yearly forecasting support
      - Dual forecasting: "CA HT NET" (sales) and "Qté Vendu" (quantity)
      - Detailed metrics for each forecasting method
      - Per-article model and forecast caching
      - Fast-mode heuristics for small series
    """

    def __init__(self, dataframe: pd.DataFrame,
                 cache_dir: str = "cache",
                 ref_col: str = "Ref Article",
                 date_col: str = "Année",
                 sales_col: str = "CA HT NET",
                 quantity_col: str = "Qté Vendu",
                 frequency: str = "yearly"):
        """
        dataframe: raw dataframe containing at least [ref_col, date_col, sales_col, quantity_col]
        cache_dir: folder to store cached models/forecasts/summaries
        frequency: 'yearly' or 'monthly' for aggregation level
        sales_col: column name for sales (default: "CA HT NET")
        quantity_col: column name for quantity (default: "Qté Vendu")
        """
        self.df_raw = dataframe.copy()
        self.ref_col = ref_col
        self.date_col = date_col
        self.sales_col = sales_col
        self.quantity_col = quantity_col
        self.frequency = frequency.lower()

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

    def clean_data(self):
        """Remove rows where sales or quantity are null/zero and keep relevant columns"""
        df = self.df_raw
        required = [self.ref_col, self.sales_col, self.quantity_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in the dataset: {missing}")

        # Keep rows where EITHER sales or quantity is non-zero
        df_clean = df[(df[self.sales_col].notna() | df[self.quantity_col].notna()) &
                      ((df[self.sales_col] != 0) | (df[self.quantity_col] != 0))].copy()
        
        if self.frequency == 'yearly':
            if 'Année' not in df_clean.columns:
                raise ValueError("'Année' column not found for yearly frequency")
            df_clean['period_key'] = df_clean['Année'].astype(int)
        else:  # monthly
            if 'Date' not in df_clean.columns:
                raise ValueError("'Date' column not found for monthly frequency")
            if not pd.api.types.is_datetime64_any_dtype(df_clean['Date']):
                df_clean['Date'] = pd.to_datetime(df_clean['Date'], errors='coerce')
            df_clean = df_clean[df_clean['Date'].notna()]
            df_clean['period_key'] = df_clean['Date'].dt.to_period('M')
            
        self.df_clean = df_clean
        return self.df_clean

    def prepare_data(self):
        """Group by article and period, summing both sales and quantities"""
        if self.df_clean is None:
            self.clean_data()

        self.df_clean[self.ref_col] = self.df_clean[self.ref_col].astype(str)

        available_columns = [c for c in ['Marque', 'Famille', 'Sous Famille', 'Designation'] 
                           if c in self.df_clean.columns]
        agg_dict = {
            self.sales_col: 'sum',
            self.quantity_col: 'sum'
        }
        for col in available_columns:
            agg_dict[col] = 'first'

        grouped = self.df_clean.groupby([self.ref_col, 'period_key']).agg(agg_dict).reset_index()
        grouped.rename(columns={'period_key': 'period'}, inplace=True)
        grouped[self.ref_col] = grouped[self.ref_col].astype(str)
        grouped = grouped.sort_values([self.ref_col, 'period']).reset_index(drop=True)

        self.grouped_data = grouped
        return grouped

    # -------------------------
    # Metric calculation utilities
    # -------------------------
    def calculate_metrics(self, actual, predicted):
        """Calculate comprehensive metrics"""
        actual = np.array(actual)
        predicted = np.array(predicted)
        
        if len(actual) == 0 or len(predicted) == 0:
            return None
            
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)
        
        mask = actual != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
            mape = np.nan
            
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
    # Forecasting methods
    # -------------------------
    def simple_moving_average(self, values, period=3, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        
        if len(values) < period:
            forecast = float(np.mean(values))
        else:
            forecast = float(np.mean(values[-period:]))
        
        if return_metrics and len(values) > period:
            predictions = [np.mean(values[i-period:i]) for i in range(period, len(values))]
            metrics = self.calculate_metrics(values[period:], predictions)
        else:
            metrics = None
            
        return (forecast, metrics) if return_metrics else forecast

    def exponential_smoothing(self, values, alpha=0.3, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        if len(values) == 1:
            return (float(values[0]), None) if return_metrics else float(values[0])
        
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
        
        if self.frequency == 'monthly':
            X = np.array([p.ordinal for p in periods]).reshape(-1, 1)
            next_X = np.array([[periods[-1].ordinal + 1]])
        else:
            X = np.array([int(p) for p in periods]).reshape(-1, 1)
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

    def arima_forecast(self, values, order=(1, 1, 0), return_metrics=False):
        if len(values) == 0 or not self.has_arima:
            return (0.0, None) if return_metrics else 0.0
        try:
            model = ARIMA(values, order=order)
            fit = model.fit()
            forecast = float(fit.forecast(steps=1)[0])
            forecast = max(0.0, forecast)
            
            if return_metrics and len(values) > 3:
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
            if self.frequency == 'monthly':
                dates = [p.to_timestamp() for p in periods]
            else:
                dates = pd.to_datetime([str(int(p)) for p in periods], format='%Y')
            
            dfp = pd.DataFrame({'ds': dates, 'y': values})
            m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            m.fit(dfp)
            
            freq = 'MS' if self.frequency == 'monthly' else 'YS'
            future = m.make_future_dataframe(periods=1, freq=freq)
            fc = m.predict(future)
            forecast = float(max(0.0, fc.iloc[-1]['yhat']))
            
            if return_metrics:
                in_sample = m.predict(dfp)
                metrics = self.calculate_metrics(values, in_sample['yhat'].values)
            else:
                metrics = None
                
            return (forecast, metrics) if return_metrics else forecast
        except Exception:
            return (0.0, None) if return_metrics else 0.0

    def xgboost_forecast(self, periods, values, return_metrics=False):
        if len(values) == 0:
            return (0.0, None) if return_metrics else 0.0
        
        if self.frequency == 'monthly':
            X = np.array([p.ordinal for p in periods]).reshape(-1, 1)
            next_X = np.array([[periods[-1].ordinal + 1]])
        else:
            X = np.array([int(p) for p in periods]).reshape(-1, 1)
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
    def _forecast_cache_path(self, ref_article):
        safe = str(ref_article).replace("/", "_").replace(" ", "_")
        return self.forecast_cache / f"{safe}__forecast.csv"

    def _summary_cache_path(self):
        return self.summary_cache / "summary.parquet"

    def _get_article_series(self, ref_article):
        """Return sorted (periods, sales_values, qty_values) arrays for the article"""
        if self.grouped_data is None:
            self.prepare_data()
        df = self.grouped_data[self.grouped_data[self.ref_col] == ref_article].sort_values('period')
        if df.empty:
            return [], [], [], {}
        
        periods = df['period'].tolist()
        sales = df[self.sales_col].astype(float).tolist()
        quantities = df[self.quantity_col].astype(float).tolist()
        
        metadata = {}
        for c in ['Designation', 'Marque', 'Famille']:
            if c in df.columns:
                metadata[c.lower()] = df[c].iloc[0]
            else:
                metadata[c.lower()] = None
        return periods, sales, quantities, metadata

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
    # Main per-article forecast method
    # -------------------------
    def forecast_article(self, ref_article, period=3, alpha=0.3,
                         force_recompute=False, include_methods=None,
                         fast_mode=True, return_metrics=True):
        """
        Forecast for a single article - BOTH sales and quantities
        """
        import json
        
        if include_methods is None:
            include_methods = ['SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST']

        cached = None if force_recompute else self._load_forecast_cache(ref_article, use_cache=True)
        if cached is not None:
            return cached.to_dict(orient='records')[0] if not cached.empty else None

        periods, sales_values, qty_values, meta = self._get_article_series(ref_article)
        if len(sales_values) == 0 and len(qty_values) == 0:
            return None

        # Determine next period
        if self.frequency == 'monthly':
            next_period = str(periods[-1] + 1)
        else:
            next_period = int(max(periods) + 1)

        # Fast-mode heuristics
        if fast_mode and len(sales_values) < 6:
            allowed = [m for m in include_methods if m in ['SMA', 'ExpSmoothing', 'LinearReg', 'XGBOOST']]
            include_methods = allowed

        # ===== SALES FORECASTING =====
        sales_results = {}
        sales_metrics_dict = {}
        
        if len(sales_values) > 0:
            if 'SMA' in include_methods:
                forecast, metrics = self.simple_moving_average(sales_values, period, return_metrics=True)
                sales_results['sma_forecast'] = forecast
                sales_metrics_dict['sma_metrics'] = metrics
            
            if 'ExpSmoothing' in include_methods:
                forecast, metrics = self.exponential_smoothing(sales_values, alpha, return_metrics=True)
                sales_results['es_forecast'] = forecast
                sales_metrics_dict['es_metrics'] = metrics
            
            if 'LinearReg' in include_methods:
                forecast, metrics = self.linear_regression_forecast(periods, sales_values, return_metrics=True)
                sales_results['lr_forecast'] = forecast
                sales_metrics_dict['lr_metrics'] = metrics
            
            if 'ARIMA' in include_methods and self.has_arima and len(sales_values) >= 3:
                forecast, metrics = self.arima_forecast(sales_values, return_metrics=True)
                sales_results['arima_forecast'] = forecast
                sales_metrics_dict['arima_metrics'] = metrics
            
            if 'PROPHET' in include_methods and self.has_prophet and len(sales_values) >= 3:
                forecast, metrics = self.prophet_forecast(periods, sales_values, return_metrics=True)
                sales_results['prophet_forecast'] = forecast
                sales_metrics_dict['prophet_metrics'] = metrics
            
            if 'XGBOOST' in include_methods:
                forecast, metrics = self.xgboost_forecast(periods, sales_values, return_metrics=True)
                sales_results['xgb_forecast'] = forecast
                sales_metrics_dict['xgb_metrics'] = metrics
        
        sales_method_values = [v for v in sales_results.values() if v is not None and not np.isnan(v)]
        avg_sales_forecast = float(np.mean(sales_method_values)) if len(sales_method_values) > 0 else 0.0

        # ===== QUANTITY FORECASTING =====
        qty_results = {}
        qty_metrics_dict = {}
        
        if len(qty_values) > 0:
            if 'SMA' in include_methods:
                forecast, metrics = self.simple_moving_average(qty_values, period, return_metrics=True)
                qty_results['sma_forecast'] = forecast
                qty_metrics_dict['sma_metrics'] = metrics
            
            if 'ExpSmoothing' in include_methods:
                forecast, metrics = self.exponential_smoothing(qty_values, alpha, return_metrics=True)
                qty_results['es_forecast'] = forecast
                qty_metrics_dict['es_metrics'] = metrics
            
            if 'LinearReg' in include_methods:
                forecast, metrics = self.linear_regression_forecast(periods, qty_values, return_metrics=True)
                qty_results['lr_forecast'] = forecast
                qty_metrics_dict['lr_metrics'] = metrics
            
            if 'ARIMA' in include_methods and self.has_arima and len(qty_values) >= 3:
                forecast, metrics = self.arima_forecast(qty_values, return_metrics=True)
                qty_results['arima_forecast'] = forecast
                qty_metrics_dict['arima_metrics'] = metrics
            
            if 'PROPHET' in include_methods and self.has_prophet and len(qty_values) >= 3:
                forecast, metrics = self.prophet_forecast(periods, qty_values, return_metrics=True)
                qty_results['prophet_forecast'] = forecast
                qty_metrics_dict['prophet_metrics'] = metrics
            
            if 'XGBOOST' in include_methods:
                forecast, metrics = self.xgboost_forecast(periods, qty_values, return_metrics=True)
                qty_results['xgb_forecast'] = forecast
                qty_metrics_dict['xgb_metrics'] = metrics
        
        qty_method_values = [v for v in qty_results.values() if v is not None and not np.isnan(v)]
        avg_qty_forecast = float(np.mean(qty_method_values)) if len(qty_method_values) > 0 else 0.0

        # Build result dictionary
        result = {
            'ref_article': ref_article,
            'designation': meta.get('designation'),
            'marque': meta.get('marque'),
            'famille': meta.get('famille'),
            'frequency': self.frequency,
            'next_period': next_period,
            
            # Historical data
            'historical_periods': json.dumps([str(p) for p in periods]),
            'historical_sales': json.dumps([float(v) for v in sales_values]),
            'historical_quantities': json.dumps([float(v) for v in qty_values]),
            
            # Sales forecasts
            'sales_avg_forecast': float(avg_sales_forecast),
            'sales_avg': float(np.mean(sales_values)) if len(sales_values) > 0 else 0.0,
            'sales_max': float(np.max(sales_values)) if len(sales_values) > 0 else 0.0,
            'sales_min': float(np.min(sales_values)) if len(sales_values) > 0 else 0.0,
            'sales_std': float(np.std(sales_values)) if len(sales_values) > 0 else 0.0,
            'sales_trend_pct': float(((sales_values[-1] - sales_values[0]) / sales_values[0]) * 100) if len(sales_values) > 0 and sales_values[0] != 0 else 0.0,
            
            # Quantity forecasts
            'qty_avg_forecast': float(avg_qty_forecast),
            'qty_avg': float(np.mean(qty_values)) if len(qty_values) > 0 else 0.0,
            'qty_max': float(np.max(qty_values)) if len(qty_values) > 0 else 0.0,
            'qty_min': float(np.min(qty_values)) if len(qty_values) > 0 else 0.0,
            'qty_std': float(np.std(qty_values)) if len(qty_values) > 0 else 0.0,
            'qty_trend_pct': float(((qty_values[-1] - qty_values[0]) / qty_values[0]) * 100) if len(qty_values) > 0 and qty_values[0] != 0 else 0.0,
            
            'data_points': len(sales_values)
        }
        
        # Add individual sales forecasts and metrics
        for key, val in sales_results.items():
            result[f'sales_{key}'] = float(val) if val is not None else np.nan
        for key, metrics in sales_metrics_dict.items():
            result[f'sales_{key}'] = json.dumps(metrics) if metrics else None
        
        # Add individual quantity forecasts and metrics
        for key, val in qty_results.items():
            result[f'qty_{key}'] = float(val) if val is not None else np.nan
        for key, metrics in qty_metrics_dict.items():
            result[f'qty_{key}'] = json.dumps(metrics) if metrics else None

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
            df_all = df_all.sort_values('sales_avg_forecast', ascending=False).reset_index(drop=True)

        summary_path = self._summary_cache_path()
        df_all.to_parquet(summary_path, index=False)

        self.forecast_results = df_all
        return df_all

    # -------------------------
    # Utility methods
    # -------------------------
    def generate_summary(self, force_recompute=False):
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
            recs.append(row)

        df_summary = pd.DataFrame(recs)
        if not df_summary.empty:
            df_summary['ref_article'] = df_summary['ref_article'].astype(str)
            df_summary = df_summary.sort_values('sales_avg_forecast', ascending=False).reset_index(drop=True)
            df_summary.to_parquet(summary_path, index=False)
        self.forecast_results = df_summary
        return df_summary

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
        self.clear_forecast_cache()
        self.clear_summary_cache()