import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


class SalesForecaster:
    """
    Generic Sales Forecasting System for predicting CA HT NET by Article Reference
    
    Features:
    - Multiple forecasting methods (SMA, Exponential Smoothing, Linear Regression)
    - Automatic data cleaning (removes CA HT NET = 0)
    - Detailed statistics and trend analysis
    - Visualization capabilities
    - Export results to CSV
    """
    
    def __init__(self, data_path=None, dataframe=None):
        """
        Initialize forecaster with data
        
        Args:
            data_path: Path to CSV file
            dataframe: Pandas DataFrame (alternative to data_path)
        """
        if data_path:
            self.df = pd.read_csv(data_path)
        elif dataframe is not None:
            self.df = dataframe.copy()
        else:
            raise ValueError("Provide either data_path or dataframe")
        
        self.df_clean = None
        self.grouped_data = None
        self.forecast_results = {}
        
    def clean_data(self):
        """Clean data by removing rows where CA HT NET = 0"""
        print("🔍 Cleaning data...")
        initial_count = len(self.df)
        
        # Remove rows where CA HT NET is 0 or null
        self.df_clean = self.df[
            (self.df['CA HT NET'].notna()) & 
            (self.df['CA HT NET'] != 0)
        ].copy()
        
        removed_count = initial_count - len(self.df_clean)
        print(f"✅ Removed {removed_count} rows with CA HT NET = 0 or null")
        print(f"✅ Clean dataset: {len(self.df_clean)} records")
        
        return self.df_clean
    
    def prepare_data(self):
        """Group data by article and year"""
        if self.df_clean is None:
            self.clean_data()
        
        print("\n📊 Preparing data by article...")

        # Handle datasets that may use "Intitule ..." instead of simple names
        rename_map = {
            'Intitule Marque': 'Marque',
            'Intitule Famille': 'Famille',
            'Intitule Sous Famille': 'Sous Famille'
        }

        # Rename columns only if they exist in your DataFrame
        for old, new in rename_map.items():
            if old in self.df_clean.columns and new not in self.df_clean.columns:
                self.df_clean.rename(columns={old: new}, inplace=True)

        # Group by article and year, sum CA HT NET
        available_columns = [c for c in ['Marque', 'Famille', 'Sous Famille', 'Designation'] if c in self.df_clean.columns]
        
        agg_dict = {'CA HT NET': 'sum'}
        for col in available_columns:
            agg_dict[col] = 'first'

        self.grouped_data = self.df_clean.groupby(
            ['Ref Article', 'Année']
        ).agg(agg_dict).reset_index()

        print(f"✅ Found {self.grouped_data['Ref Article'].nunique()} unique articles")

        return self.grouped_data

    
    def simple_moving_average(self, values, period=3):
        """Calculate Simple Moving Average"""
        if len(values) < period:
            return np.mean(values) if len(values) > 0 else 0
        return np.mean(values[-period:])
    
    def exponential_smoothing(self, values, alpha=0.3):
        """Calculate Exponential Smoothing forecast"""
        if len(values) == 0:
            return 0
        if len(values) == 1:
            return values[0]
        
        forecast = values[0]
        for val in values[1:]:
            forecast = alpha * val + (1 - alpha) * forecast
        
        return forecast
    
    def linear_regression_forecast(self, years, values):
        """Calculate Linear Regression forecast"""
        if len(years) == 0:
            return 0
        
        X = np.array(years).reshape(-1, 1)
        y = np.array(values)
        
        model = LinearRegression()
        model.fit(X, y)
        
        next_year = max(years) + 1
        forecast = model.predict([[next_year]])[0]
        
        return max(0, forecast)  # Ensure non-negative
    

    def arima_forecast(self, values, order=(1, 1, 0)):
        """
        Forecast using ARIMA model.
        
        Args:
            values: list or array of historical sales
            order: ARIMA parameters (p, d, q)
            
        Returns:
            forecast for the next period
        """
        if len(values) == 0:
            return 0
        try:
            model = ARIMA(values, order=order)
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)[0]
            return max(0, forecast)  # Ensure non-negative
        except Exception as e:
            print("⚠️ ARIMA forecast failed:", e)
            return 0

    def prophet_forecast(self, years, values):
        """
        Forecast using Prophet model.

        Args:
            years: list of years (ints)
            values: list of historical sales

        Returns:
            Forecast for the next year (float)
        """
        if len(values) < 2:
            return 0

        try:
            # Prophet expects columns ['ds', 'y']
            df_prophet = pd.DataFrame({
                'ds': pd.to_datetime(years, format='%Y'),
                'y': values
            })
            model = Prophet(yearly_seasonality=True)
            model.fit(df_prophet)

            # Predict one year into the future
            future = model.make_future_dataframe(periods=1, freq='Y')
            forecast = model.predict(future)

            next_value = forecast.iloc[-1]['yhat']
            return max(0, next_value)
        except Exception as e:
            print("⚠️ Prophet forecast failed:", e)
            return 0





    def forecast_article(self, ref_article, period=3, alpha=0.3):
        """
        Forecast CA HT NET for a specific article
        
        Args:
            ref_article: Reference of the article to forecast
            period: Period for Simple Moving Average
            alpha: Smoothing factor for Exponential Smoothing
            
        Returns:
            Dictionary with forecast results
        """
        if self.grouped_data is None:
            self.prepare_data()
        
        # Get data for specific article
        article_data = self.grouped_data[
            self.grouped_data['Ref Article'] == ref_article
        ].sort_values('Année')
        
        if len(article_data) == 0:
            return None
        
        years = article_data['Année'].values
        values = article_data['CA HT NET'].values
        
        # Calculate forecasts
        next_year = int(max(years) + 1)
        
        sma_forecast = self.simple_moving_average(values, period)
        es_forecast = self.exponential_smoothing(values, alpha)
        lr_forecast = self.linear_regression_forecast(years, values)
        arima_forecast = self.arima_forecast(values)
        prophet_forecast = self.prophet_forecast(years, values)

        
        # Average of all methods
        avg_forecast = np.mean([sma_forecast, es_forecast, lr_forecast, arima_forecast])
        
        # Calculate statistics
        avg_sales = np.mean(values)
        max_sales = np.max(values)
        min_sales = np.min(values)
        std_sales = np.std(values)
        
        # Calculate trend
        if len(values) > 1:
            trend_pct = ((values[-1] - values[0]) / values[0]) * 100
        else:
            trend_pct = 0
        
        # Store results
        result = {
            'ref_article': ref_article,
            'designation': article_data['Designation'].iloc[0],
            'marque': article_data['Marque'].iloc[0],
            'famille': article_data['Famille'].iloc[0],
            'next_year': next_year,
            'sma_forecast': sma_forecast,
            'es_forecast': es_forecast,
            'lr_forecast': lr_forecast,
            'arima_forecast': arima_forecast,
            'avg_forecast': avg_forecast,
            'historical_years': years.tolist(),
            'historical_values': values.tolist(),
            'avg_sales': avg_sales,
            'max_sales': max_sales,
            'min_sales': min_sales,
            'std_sales': std_sales,
            'trend_pct': trend_pct,
            'data_points': len(values)
        }
        
        return result
    
    def forecast_all_articles(self, period=3, alpha=0.3):
        """
        Forecast CA HT NET for all articles
        
        Args:
            period: Period for Simple Moving Average
            alpha: Smoothing factor for Exponential Smoothing
            
        Returns:
            DataFrame with all forecasts
        """
        if self.grouped_data is None:
            self.prepare_data()
        
        print("\n🔮 Forecasting all articles...")
        
        all_forecasts = []
        articles = self.grouped_data['Ref Article'].unique()
        
        for i, ref_article in enumerate(articles, 1):
            if i % 50 == 0:
                print(f"   Processing {i}/{len(articles)} articles...")
            
            result = self.forecast_article(ref_article, period, alpha)
            if result:
                all_forecasts.append(result)
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(all_forecasts)
        
        # Sort by forecast value (descending)
        forecast_df = forecast_df.sort_values('avg_forecast', ascending=False)
        
        self.forecast_results = forecast_df
        
        print(f"✅ Forecasted {len(forecast_df)} articles")
        
        return forecast_df
    
    def visualize_article(self, ref_article, figsize=(12, 6)):
        """
        Visualize historical data and forecast for a specific article
        
        Args:
            ref_article: Reference of the article
            figsize: Figure size (width, height)
        """
        result = self.forecast_article(ref_article)
        
        if result is None:
            print(f"❌ No data found for article: {ref_article}")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Historical + Forecast
        years = result['historical_years']
        values = result['historical_values']
        
        axes[0].plot(years, values, 'o-', linewidth=2, markersize=8, 
                     label='Historical', color='#3b82f6')
        axes[0].plot([result['next_year']], [result['avg_forecast']], 
                     'o', markersize=10, label='Forecast', color='#10b981')
        axes[0].axhline(y=result['avg_sales'], color='#ef4444', 
                       linestyle='--', label='Avg Sales')
        
        axes[0].set_xlabel('Year', fontsize=12)
        axes[0].set_ylabel('CA HT NET (DT)', fontsize=12)
        axes[0].set_title(f'{ref_article}\n{result["designation"]}', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Forecast Methods Comparison
        methods = ['SMA', 'Exp. Smoothing', 'Linear Reg.', 'ARIMA', 'Average']
        forecasts = [
            result['sma_forecast'],
            result['es_forecast'],
            result['lr_forecast'],
            result['arima_forecast'],
            result['avg_forecast']
        ]
        colors = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ec4899', '#10b981']
        bars = axes[1].bar(methods, forecasts, color=colors, alpha=0.8)
        
        axes[1].set_ylabel('Forecast CA HT NET (DT)', fontsize=12)
        axes[1].set_title(f'Forecast Comparison ({result["next_year"]})', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\n📈 Statistics for {ref_article}")
        print(f"   Designation: {result['designation']}")
        print(f"   Brand: {result['marque']} | Family: {result['famille']}")
        print(f"   Data Points: {result['data_points']} years")
        print(f"   Avg Sales: {result['avg_sales']:.2f} DT")
        print(f"   Trend: {result['trend_pct']:+.2f}%")
        print(f"   Forecast ({result['next_year']}): {result['avg_forecast']:.2f} DT")
    
    def get_top_articles(self, n=10, metric='avg_forecast'):
        """
        Get top N articles by specified metric
        
        Args:
            n: Number of top articles
            metric: Metric to sort by (avg_forecast, trend_pct, avg_sales, etc.)
        """
        if self.forecast_results is None or len(self.forecast_results) == 0:
            self.forecast_all_articles()
        
        ascending = False if metric in ['avg_forecast', 'avg_sales', 'trend_pct'] else True
        
        top = self.forecast_results.sort_values(metric, ascending=ascending).head(n)
        
        return top[['ref_article', 'designation', 'marque', metric, 'next_year']]
    
    def export_results(self, filename='forecast_results.csv'):
        """Export forecast results to CSV"""
        if self.forecast_results is None or len(self.forecast_results) == 0:
            print("❌ No forecast results to export. Run forecast_all_articles() first.")
            return
        
        # Select columns for export
        export_df = self.forecast_results[[
            'ref_article', 'designation', 'marque', 'famille',
            'next_year', 'avg_forecast', 'sma_forecast', 
            'es_forecast', 'lr_forecast',
            'avg_sales', 'trend_pct', 'data_points'
        ]].copy()
        
        export_df.to_csv(filename, index=False)
        print(f"✅ Results exported to {filename}")
    
    def summary_report(self):
        """Generate summary report"""
        if self.forecast_results is None or len(self.forecast_results) == 0:
            self.forecast_all_articles()
        
        print("\n" + "="*60)
        print("📊 FORECAST SUMMARY REPORT")
        print("="*60)
        
        print(f"\n🔢 Total Articles Forecasted: {len(self.forecast_results)}")
        print(f"📅 Forecast Year: {int(self.forecast_results['next_year'].iloc[0])}")
        
        print(f"\n💰 Total Forecast CA HT NET: {self.forecast_results['avg_forecast'].sum():.2f} DT")
        print(f"📈 Average Forecast per Article: {self.forecast_results['avg_forecast'].mean():.2f} DT")
        
        print(f"\n📊 Trend Analysis:")
        positive_trend = (self.forecast_results['trend_pct'] > 0).sum()
        negative_trend = (self.forecast_results['trend_pct'] < 0).sum()
        print(f"   ↗️  Positive Trend: {positive_trend} articles ({positive_trend/len(self.forecast_results)*100:.1f}%)")
        print(f"   ↘️  Negative Trend: {negative_trend} articles ({negative_trend/len(self.forecast_results)*100:.1f}%)")
        
        print(f"\n🏆 Top 5 Articles by Forecast:")
        top5 = self.forecast_results.head(5)
        for i, row in top5.iterrows():
            print(f"   {row['ref_article']}: {row['avg_forecast']:.2f} DT ({row['trend_pct']:+.1f}%)")
        
        print("\n" + "="*60)


