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
    SalesForecaster (refactored) with:
      - Multiple forecast methods (SMA, Exp Smoothing, Linear Regression, ARIMA, Prophet, XGBoost)
      - Per-article model and forecast caching
      - Fast-mode heuristics for small series
      - Summary / trend caching for dashboard
    """

    def __init__(self, dataframe: pd.DataFrame,
                 cache_dir: str = "cache",
                 ref_col: str = "Ref Article",
                 year_col: str = "Année",
                 sales_col: str = "CA HT NET"):
        """
        dataframe: raw dataframe containing at least [ref_col, year_col, sales_col]
        cache_dir: folder to store cached models/forecasts/summaries
        """
        self.df_raw = dataframe.copy()
        self.ref_col = ref_col
        self.year_col = year_col
        self.sales_col = sales_col

        # ensure standard columns rename when necessary
        self._normalize_column_names()

        self.df_clean = None
        self.grouped_data = None
        self.forecast_results = None  # pandas DataFrame with forecasts summary

        # cache structure
        self.cache_dir = Path(cache_dir)
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
    # Data prep
    # -------------------------
    def clean_data(self):
        """Remove rows where sales are null or zero and keep relevant columns"""
        df = self.df_raw
        required = [self.ref_col, self.year_col, self.sales_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in the dataset: {missing}")

        df_clean = df[df[self.sales_col].notna() & (df[self.sales_col] != 0)].copy()
        # ensure year is integer
        df_clean[self.year_col] = df_clean[self.year_col].astype(int)
        self.df_clean = df_clean
        return self.df_clean

    def prepare_data(self):
        """Group by article and year summing sales and keep representative metadata"""
        if self.df_clean is None:
            self.clean_data()

        # Ensure reference column is string to avoid Parquet serialization issues later
        self.df_clean[self.ref_col] = self.df_clean[self.ref_col].astype(str)

        available_columns = [c for c in ['Marque', 'Famille', 'Sous Famille', 'Designation'] if c in self.df_clean.columns]
        agg_dict = {self.sales_col: 'sum'}
        for col in available_columns:
            agg_dict[col] = 'first'

        grouped = self.df_clean.groupby([self.ref_col, self.year_col]).agg(agg_dict).reset_index()

        # Also ensure the grouped version keeps ref_article as string
        grouped[self.ref_col] = grouped[self.ref_col].astype(str)

        self.grouped_data = grouped
        return grouped


    # -------------------------
    # Low-cost forecasting helpers
    # -------------------------
    def simple_moving_average(self, values, period=3):
        if len(values) == 0:
            return 0.0
        if len(values) < period:
            return float(np.mean(values))
        return float(np.mean(values[-period:]))

    def exponential_smoothing(self, values, alpha=0.3):
        if len(values) == 0:
            return 0.0
        if len(values) == 1:
            return float(values[0])
        f = values[0]
        for v in values[1:]:
            f = alpha * v + (1 - alpha) * f
        return float(f)

    def linear_regression_forecast(self, years, values):
        if len(years) == 0:
            return 0.0
        X = np.array(years).reshape(-1, 1)
        y = np.array(values)
        model = LinearRegression()
        model.fit(X, y)
        next_year = int(max(years) + 1)
        pred = float(model.predict(np.array([[next_year]]))[0])
        return max(0.0, pred)

    # -------------------------
    # Time series / ML forecasts
    # -------------------------
    def arima_forecast(self, values, order=(1, 1, 0)):
        if len(values) == 0 or not self.has_arima:
            return 0.0
        try:
            model = ARIMA(values, order=order)
            fit = model.fit()
            pred = float(fit.forecast(steps=1)[0])
            return max(0.0, pred)
        except Exception:
            return 0.0

    def prophet_forecast(self, years, values):
        if len(values) < 2 or not self.has_prophet:
            return 0.0
        try:
            dfp = pd.DataFrame({'ds': pd.to_datetime(years, format='%Y'), 'y': values})
            m = Prophet(yearly_seasonality=True)
            m.fit(dfp)
            future = m.make_future_dataframe(periods=1, freq='Y')
            fc = m.predict(future)
            return float(max(0.0, fc.iloc[-1]['yhat']))
        except Exception:
            return 0.0

    def xgboost_forecast(self, years, values):
        """
        Use XGBoost if available, else fall back to RandomForestRegressor.
        We treat years as numeric features.
        """
        if len(values) == 0:
            return 0.0
        X = np.array(years).reshape(-1, 1)
        y = np.array(values)
        try:
            if self.has_xgboost:
                model = XGBRegressor(n_estimators=200, learning_rate=0.05, verbosity=0, n_jobs=1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            # train on all data (small series)
            model.fit(X, y)
            next_year = np.array([[int(max(years) + 1)]])
            pred = float(model.predict(next_year)[0])
            return max(0.0, pred)
        except Exception:
            return 0.0

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
        """Return sorted (years, values) arrays for the article"""
        if self.grouped_data is None:
            self.prepare_data()
        df = self.grouped_data[self.grouped_data[self.ref_col] == ref_article].sort_values(self.year_col)
        if df.empty:
            return [], []
        years = df[self.year_col].astype(int).tolist()
        values = df[self.sales_col].astype(float).tolist()
        # find representative metadata
        metadata = {}
        for c in ['Designation', 'Marque', 'Famille']:
            if c in df.columns:
                metadata[c.lower()] = df[c].iloc[0]
            else:
                metadata[c.lower()] = None
        return years, values, metadata

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
    # Main per-article forecast method (caching + fast heuristics)
    # -------------------------
    def forecast_article(self, ref_article, period=3, alpha=0.3,
                         force_recompute=False, include_methods=None,
                         fast_mode=True):
        """
        Forecast for a single article. Uses caching for models & forecasts.
        Args:
            ref_article: article id
            period, alpha: used by sma and exp smoothing
            force_recompute: bypass caches if True
            include_methods: list of method names to compute (SMA, ES, LR, ARIMA, PROPHET, XGBOOST)
            fast_mode: if True, skip ARIMA/PROPHET for short series (<4 points)
        Returns:
            dict with forecast fields and stats
        """
        if include_methods is None:
            include_methods = ['SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST']

        cached = None if force_recompute else self._load_forecast_cache(ref_article, use_cache=True)
        if cached is not None:
            # return cached dict
            return cached.to_dict(orient='records')[0] if not cached.empty else None

        years, values, meta = self._get_article_series(ref_article)
        if len(values) == 0:
            return None

        next_year = int(max(years) + 1)

        # Fast-mode heuristics: skip heavy methods for short history
        if fast_mode and len(values) < 6:
            allowed = []
            for m in include_methods:
                if m in ['SMA', 'ExpSmoothing', 'LinearReg', 'XGBOOST']:
                    allowed.append(m)
            include_methods = allowed

        # compute each method if requested
        sma = self.simple_moving_average(values, period) if 'SMA' in include_methods else None
        es = self.exponential_smoothing(values, alpha) if 'ExpSmoothing' in include_methods else None
        lr = self.linear_regression_forecast(years, values) if 'LinearReg' in include_methods else None

        arima = None
        if 'ARIMA' in include_methods and self.has_arima and len(values) >= 3:
            arima = self.arima_forecast(values)

        prophet = None
        if 'PROPHET' in include_methods and self.has_prophet and len(values) >= 3:
            prophet = self.prophet_forecast(years, values)

        xgb = None
        if 'XGBOOST' in include_methods:
            xgb = self.xgboost_forecast(years, values)

        # build ensemble average (only non-None numeric methods)
        method_values = [v for v in [sma, es, lr, arima, prophet, xgb] if v is not None]
        avg_forecast = float(np.mean(method_values)) if len(method_values) > 0 else 0.0

        # stats
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
            'next_year': next_year,
            'sma_forecast': float(sma) if sma is not None else np.nan,
            'es_forecast': float(es) if es is not None else np.nan,
            'lr_forecast': float(lr) if lr is not None else np.nan,
            'arima_forecast': float(arima) if arima is not None else np.nan,
            'prophet_forecast': float(prophet) if prophet is not None else np.nan,
            'xgb_forecast': float(xgb) if xgb is not None else np.nan,
            'avg_forecast': float(avg_forecast),
            'historical_years': years,
            'historical_values': values,
            'avg_sales': avg_sales,
            'max_sales': max_sales,
            'min_sales': min_sales,
            'std_sales': std_sales,
            'trend_pct': trend_pct,
            'data_points': len(values)
        }

        # save cache (single-row csv)
        df_out = pd.DataFrame([result])
        self._save_forecast_cache(ref_article, df_out)

        return result

    # -------------------------
    # Forecast all articles (uses caching & progress callback)
    # -------------------------
    def forecast_all_articles(self, period=3, alpha=0.3, force_recompute=False,
                              include_methods=None, fast_mode=True, progress_callback=None):
        """
        Loop through all articles, using cached forecasts when possible.
        progress_callback: optional function taking (i, total) to report progress
        """
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
            df_all['ref_article'] = df_all['ref_article'].astype(str)  # ✅ Fix here
            df_all = df_all.sort_values('avg_forecast', ascending=False).reset_index(drop=True)

        summary_path = self._summary_cache_path()
        df_all.to_parquet(summary_path, index=False)

        self.forecast_results = df_all
        return df_all

    # -------------------------
    # Trend classification & summary utilities
    # -------------------------
    def classify_trend_label(self, last_actual_mean, next_forecast_mean, tol=0.05):
        """
        Small utility to classify trend: Uptrend / Downtrend / Stable
        tol = 0.05 means +/-5% considered stable
        """
        if last_actual_mean == 0:
            return "Stable"
        change = (next_forecast_mean - last_actual_mean) / last_actual_mean
        if change > tol:
            return "Uptrend"
        elif change < -tol:
            return "Downtrend"
        else:
            return "Stable"

    def generate_summary(self, lookback_years=3, force_recompute=False):
        """
        Generate summary DataFrame with per-article metrics and trend labels.
        Uses cached per-article forecasts to compute distribution quickly.
        """
        summary_path = self._summary_cache_path()
        if (not force_recompute) and summary_path.exists():
            try:
                df = pd.read_parquet(summary_path)
                self.forecast_results = df
                return df
            except Exception:
                pass

        # else compute from scratch using per-article cached forecasts
        if self.grouped_data is None:
            self.prepare_data()

        articles = sorted(self.grouped_data[self.ref_col].unique().tolist())
        recs = []
        for a in articles:
            f = self._load_forecast_cache(a, use_cache=True)
            if f is None:
                # try to compute quickly (fast_mode) but do not use heavy methods
                res = self.forecast_article(a, fast_mode=True)
                f = pd.DataFrame([res]) if res is not None else None
            if f is None or f.empty:
                continue

            # transform f to record
            row = f.iloc[0].to_dict()

            # compute last actual mean over lookback_years
            yrs = row.get('historical_years', [])
            vals = row.get('historical_values', [])
            if len(yrs) == 0:
                continue
            # take last N years mean
            if lookback_years <= 0:
                last_mean = float(np.mean(vals))
            else:
                last_vals = vals[-lookback_years:] if len(vals) >= lookback_years else vals
                last_mean = float(np.mean(last_vals)) if last_vals else 0.0

            next_forecast = float(row.get('avg_forecast', 0.0))
            trend_label = self.classify_trend_label(last_mean, next_forecast, tol=0.05)

            recs.append({
                'ref_article': row['ref_article'],
                'designation': row.get('designation'),
                'marque': row.get('marque'),
                'famille': row.get('famille'),
                'next_year': row.get('next_year'),
                'avg_forecast': float(row.get('avg_forecast', 0.0)),
                'trend_pct': float(row.get('trend_pct', 0.0)),
                'trend_label': trend_label,
                'data_points': int(row.get('data_points', 0))
            })

        df_summary = pd.DataFrame(recs)
        if not df_summary.empty:
            df_summary['ref_article'] = df_summary['ref_article'].astype(str)  # ✅ Fix here
            df_summary = df_summary.sort_values('avg_forecast', ascending=False).reset_index(drop=True)
            df_summary.to_parquet(summary_path, index=False)
        self.forecast_results = df_summary
        return df_summary

    # -------------------------
    # Utilities to clear caches
    # -------------------------
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
