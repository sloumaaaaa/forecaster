import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import ast
import sys
from datetime import datetime

# Import your forecaster
from sales_forecaster2 import SalesForecaster

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        background-color: #d1ecf1;
        border-left: 5px solid #17a2b8;
        border-radius: 0.3rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

def load_data(uploaded_file, frequency, date_col):
    """Load and initialize the forecaster with uploaded data"""
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        
        # Initialize forecaster
        forecaster = SalesForecaster(
            dataframe=df,
            cache_dir="cache",
            ref_col="Ref Article",
            date_col=date_col,
            sales_col="CA HT NET",
            frequency=frequency
        )
        
        return forecaster
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def parse_json_field(value):
    """Parse JSON string or Python string representation, or return the value as-is if it's already a list/dict"""
    if value is None or pd.isna(value):
        return None
    
    # If it's already a list or dict, return it
    if isinstance(value, (list, dict)):
        return value
    
    # If it's a string, try to parse it
    if isinstance(value, str):
        # First try JSON (with double quotes)
        try:
            return json.loads(value)
        except:
            pass
        
        # Then try Python literal eval (with single quotes)
        try:
            return ast.literal_eval(value)
        except:
            pass
    
    return None

def parse_metrics(metrics_str):
    """Parse metrics from JSON string"""
    return parse_json_field(metrics_str)

def create_forecast_chart(historical_periods, historical_values, forecast_value, next_period, frequency):
    """Create an interactive forecast visualization - fixed for monthly support"""
    
    # Parse historical_values if it's a JSON string
    if isinstance(historical_values, str):
        historical_values = parse_json_field(historical_values)
    
    # Parse historical_periods if it's a JSON string
    if isinstance(historical_periods, str):
        historical_periods = parse_json_field(historical_periods)
    
    # Ensure we have valid data
    if historical_values is None or historical_periods is None:
        st.error(f"Invalid historical data format. Periods: {type(historical_periods)}, Values: {type(historical_values)}")
        st.write("**Unable to parse historical data. Please check your forecaster output format.**")
        return None
    
    # Convert to lists if they're numpy arrays
    if isinstance(historical_values, np.ndarray):
        historical_values = historical_values.tolist()
    if isinstance(historical_periods, np.ndarray):
        historical_periods = historical_periods.tolist()
    
    # Convert periods to strings for display
    # For monthly, periods come as strings like '2024-01'
    # For yearly, they come as integers
    if frequency == 'monthly':
        period_labels = [str(p) for p in historical_periods]
        next_period_label = str(next_period)
    else:
        period_labels = [str(int(p)) for p in historical_periods]
        next_period_label = str(int(next_period))
    
    # Create numeric x-axis for proper ordering
    x_numeric = list(range(len(historical_periods)))
    x_numeric_next = len(historical_periods)
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=x_numeric,
        y=historical_values,
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        text=period_labels,
        hovertemplate='<b>%{text}</b><br>Sales: %{y:.2f}<extra></extra>'
    ))
    
    # Forecast point - connect last historical point to forecast
    fig.add_trace(go.Scatter(
        x=[x_numeric[-1], x_numeric_next],
        y=[historical_values[-1], forecast_value],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=10, symbol='star'),
        text=[period_labels[-1], next_period_label],
        hovertemplate='<b>%{text}</b><br>Sales: %{y:.2f}<extra></extra>'
    ))
    
    # Update x-axis to show period labels
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,
        dtick=1,
        ticktext=period_labels + [next_period_label],
        tickvals=x_numeric + [x_numeric_next]
    )
    
    fig.update_layout(
        title=f"Sales Forecast ({frequency.capitalize()})",
        xaxis_title="Period",
        yaxis_title="Sales (CA HT NET)",
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_metrics_chart(metrics_dict, method_name):
    """Create a bar chart for metrics"""
    if not metrics_dict or all(pd.isna(v) for v in metrics_dict.values()):
        return None
    
    metrics_df = pd.DataFrame({
        'Metric': list(metrics_dict.keys()),
        'Value': list(metrics_dict.values())
    })
    
    # Remove NaN values
    metrics_df = metrics_df.dropna()
    
    if metrics_df.empty:
        return None
    
    fig = px.bar(
        metrics_df,
        x='Metric',
        y='Value',
        title=f'{method_name} Performance Metrics',
        color='Value',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=300,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def display_method_comparison(result):
    """Display comparison of all forecasting methods"""
    methods = {
        'Simple Moving Average': 'sma_forecast',
        'Exponential Smoothing': 'es_forecast',
        'Linear Regression': 'lr_forecast',
        'ARIMA': 'arima_forecast',
        'Prophet': 'prophet_forecast',
        'XGBoost/Random Forest': 'xgb_forecast'
    }
    
    comparison_data = []
    for name, key in methods.items():
        if key in result and not pd.isna(result[key]):
            comparison_data.append({
                'Method': name,
                'Forecast': float(result[key])
            })
    
    if comparison_data:
        df_comp = pd.DataFrame(comparison_data)
        fig = px.bar(
            df_comp,
            x='Method',
            y='Forecast',
            title='Forecast Comparison by Method',
            color='Forecast',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, template='plotly_white')
        fig.add_hline(
            y=result['avg_forecast'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Ensemble Average: {result['avg_forecast']:.2f}"
        )
        st.plotly_chart(fig, use_container_width=True)

# Main App
st.markdown('<p class="main-header">📊 Sales Forecasting Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Sales Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your sales data in CSV or Excel format"
    )
    
    st.markdown("---")
    
    # Frequency selection
    frequency = st.radio(
        "📅 Forecasting Frequency",
        options=['yearly', 'monthly'],
        index=0,
        help="Choose between yearly or monthly forecasting"
    )
    
    # Date column based on frequency
    if frequency == 'yearly':
        date_col = st.text_input("Year Column Name", value="Année", help="Column containing year data")
    else:
        date_col = st.text_input("Date Column Name", value="Date", help="Column containing date data (must be parseable as datetime)")
    
    st.markdown("---")
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        period = st.slider("Moving Average Period", 1, 12, 3, help="Window size for moving average")
        alpha = st.slider("Exponential Smoothing Alpha", 0.1, 0.9, 0.3, 0.1, help="Smoothing parameter")
        
        st.markdown("**Forecasting Methods:**")
        include_methods = st.multiselect(
            "Select Methods",
            options=['SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST'],
            default=['SMA', 'ExpSmoothing', 'LinearReg', 'ARIMA', 'PROPHET', 'XGBOOST']
        )
        
        fast_mode = st.checkbox("Fast Mode", value=True, help="Use simplified methods for articles with <6 data points")
        force_recompute = st.checkbox("Force Recompute", value=False, help="Ignore cached results and recompute all forecasts")
    
    st.markdown("---")
    
    # Cache management
    with st.expander("🗑️ Cache Management"):
        st.write(f"**Current Frequency:** {frequency}")
        cache_path = Path("cache") / frequency
        if cache_path.exists():
            cache_files = list(cache_path.glob("*.json"))
            st.write(f"**Cache files:** {len(cache_files)}")
        else:
            st.write("**Cache directory not found**")
        
        if st.button("Clear All Caches"):
            if st.session_state.forecaster:
                st.session_state.forecaster.clear_all_caches()
                st.success("All caches cleared!")
            else:
                st.warning("No forecaster initialized")

# Main content
if uploaded_file is not None:
    # Load data button
    if not st.session_state.data_loaded or st.button("🔄 Load/Reload Data"):
        with st.spinner("Loading data..."):
            forecaster = load_data(uploaded_file, frequency, date_col)
            if forecaster:
                st.session_state.forecaster = forecaster
                st.session_state.data_loaded = True
                st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
                
                # Display data info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Records", len(forecaster.df_raw))
                with col2:
                    st.metric("Unique Articles", forecaster.df_raw['Ref Article'].nunique() if 'Ref Article' in forecaster.df_raw.columns else 0)
                with col3:
                    st.metric("Frequency", frequency.capitalize())
                with col4:
                    if forecaster.grouped_data is not None:
                        st.metric("Clean Records", len(forecaster.grouped_data))

# If data is loaded, show forecasting options
if st.session_state.data_loaded and st.session_state.forecaster:
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast All", "🔍 Single Article", "📊 Summary Statistics", "📋 Data Preview"])
    
    with tab1:
        st.header("Forecast All Articles")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🚀 Run Forecast", type="primary", use_container_width=True):
                forecaster = st.session_state.forecaster
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def progress_callback(current, total):
                    progress = current / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {current}/{total} articles ({progress*100:.1f}%)")
                
                with st.spinner("Forecasting all articles..."):
                    results = forecaster.forecast_all_articles(
                        period=period,
                        alpha=alpha,
                        force_recompute=force_recompute,
                        include_methods=include_methods,
                        fast_mode=fast_mode,
                        progress_callback=progress_callback
                    )
                    st.session_state.forecast_results = results
                
                progress_bar.empty()
                status_text.empty()
                st.success(f"✅ Forecasted {len(results)} articles successfully!")
        
        # Display results
        if st.session_state.forecast_results is not None:
            results_df = st.session_state.forecast_results
            
            st.markdown("### 📊 Forecast Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", len(results_df))
            with col2:
                st.metric("Avg Forecast", f"{results_df['avg_forecast'].mean():.2f}")
            with col3:
                st.metric("Total Forecast", f"{results_df['avg_forecast'].sum():.2f}")
            with col4:
                st.metric("Max Forecast", f"{results_df['avg_forecast'].max():.2f}")
            
            # Filters
            st.markdown("#### 🔍 Filter Results")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'marque' in results_df.columns:
                    marques = ['All'] + sorted([str(m) for m in results_df['marque'].dropna().unique()])
                    selected_marque = st.selectbox("Filter by Brand (Marque)", marques)
                else:
                    selected_marque = 'All'
            
            with col2:
                if 'famille' in results_df.columns:
                    familles = ['All'] + sorted([str(f) for f in results_df['famille'].dropna().unique()])
                    selected_famille = st.selectbox("Filter by Family (Famille)", familles)
                else:
                    selected_famille = 'All'
            
            with col3:
                min_forecast = st.number_input("Min Forecast Value", value=0.0, step=100.0)
            
            # Apply filters
            filtered_df = results_df.copy()
            if selected_marque != 'All':
                filtered_df = filtered_df[filtered_df['marque'] == selected_marque]
            if selected_famille != 'All':
                filtered_df = filtered_df[filtered_df['famille'] == selected_famille]
            filtered_df = filtered_df[filtered_df['avg_forecast'] >= min_forecast]
            
            # Display table
            st.markdown(f"**Showing {len(filtered_df)} of {len(results_df)} articles**")
            
            display_columns = ['ref_article', 'designation', 'marque', 'famille', 'next_period', 
                             'avg_forecast', 'avg_sales', 'trend_pct', 'data_points']
            available_cols = [col for col in display_columns if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_cols].head(100),
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"forecast_results_{frequency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.header("Single Article Forecast")
        
        # Article selection
        forecaster = st.session_state.forecaster
        if forecaster.grouped_data is None:
            forecaster.prepare_data()
        
        articles = sorted(forecaster.grouped_data['Ref Article'].unique().tolist())
        selected_article = st.selectbox("Select Article", articles, key="single_article")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔮 Forecast", type="primary", use_container_width=True):
                with st.spinner(f"Forecasting {selected_article}..."):
                    result = forecaster.forecast_article(
                        selected_article,
                        period=period,
                        alpha=alpha,
                        force_recompute=force_recompute,
                        include_methods=include_methods,
                        fast_mode=fast_mode,
                        return_metrics=True
                    )
                    
                    if result:
                        st.session_state.single_result = result
                        st.success("✅ Forecast completed!")
                    else:
                        st.error("❌ No data available for this article")
        
        # Display single article results
        if 'single_result' in st.session_state and st.session_state.single_result:
            result = st.session_state.single_result
            
            # Article info
            st.markdown("### 📦 Article Information")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Reference", result['ref_article'])
            with col2:
                st.metric("Designation", result.get('designation', 'N/A'))
            with col3:
                st.metric("Brand", result.get('marque', 'N/A'))
            with col4:
                st.metric("Family", result.get('famille', 'N/A'))
            
            # Forecast results
            st.markdown("### 🎯 Forecast Results")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Next Period", result['next_period'])
            with col2:
                st.metric("Ensemble Forecast", f"{result['avg_forecast']:.2f}")
            with col3:
                st.metric("Avg Historical Sales", f"{result['avg_sales']:.2f}")
            with col4:
                st.metric("Trend", f"{result['trend_pct']:.2f}%")
            
            # Historical stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Max Sales", f"{result['max_sales']:.2f}")
            with col2:
                st.metric("Min Sales", f"{result['min_sales']:.2f}")
            with col3:
                st.metric("Std Dev", f"{result['std_sales']:.2f}")
            with col4:
                st.metric("Data Points", result['data_points'])
            
            # Visualization
            st.markdown("### 📈 Forecast Visualization")
            chart = create_forecast_chart(
                result['historical_periods'],
                result['historical_values'],
                result['avg_forecast'],
                result['next_period'],
                frequency
            )
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Method comparison
            st.markdown("### 🔄 Method Comparison")
            display_method_comparison(result)
            
            # Individual method forecasts and metrics
            st.markdown("### 📊 Detailed Method Performance")
            
            methods_info = {
                'Simple Moving Average': ('sma_forecast', 'sma_metrics'),
                'Exponential Smoothing': ('es_forecast', 'es_metrics'),
                'Linear Regression': ('lr_forecast', 'lr_metrics'),
                'ARIMA': ('arima_forecast', 'arima_metrics'),
                'Prophet': ('prophet_forecast', 'prophet_metrics'),
                'XGBoost/RF': ('xgb_forecast', 'xgb_metrics')
            }
            
            for method_name, (forecast_key, metrics_key) in methods_info.items():
                if forecast_key in result and not pd.isna(result[forecast_key]):
                    with st.expander(f"📍 {method_name}"):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.metric("Forecast Value", f"{result[forecast_key]:.2f}")
                            
                            # Display metrics
                            if metrics_key in result:
                                metrics = parse_metrics(result[metrics_key])
                                if metrics:
                                    st.markdown("**Performance Metrics:**")
                                    for metric, value in metrics.items():
                                        if not pd.isna(value):
                                            st.write(f"- **{metric}:** {value:.4f}")
                        
                        with col2:
                            # Metrics chart
                            if metrics_key in result:
                                metrics = parse_metrics(result[metrics_key])
                                if metrics:
                                    chart = create_metrics_chart(metrics, method_name)
                                    if chart:
                                        st.plotly_chart(chart, use_container_width=True)
    
    with tab3:
        st.header("Summary Statistics")
        
        if st.session_state.forecast_results is not None:
            results_df = st.session_state.forecast_results
            
            # Overall statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Forecast Distribution")
                fig = px.histogram(
                    results_df,
                    x='avg_forecast',
                    nbins=50,
                    title='Distribution of Forecasts'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📈 Trend Distribution")
                fig = px.histogram(
                    results_df,
                    x='trend_pct',
                    nbins=50,
                    title='Distribution of Trends (%)'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Top performers
            st.markdown("### 🏆 Top 10 Articles by Forecast")
            top10 = results_df.nlargest(10, 'avg_forecast')
            fig = px.bar(
                top10,
                x='ref_article',
                y='avg_forecast',
                title='Top 10 Articles by Forecast Value',
                color='avg_forecast',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Brand/Family analysis
            if 'marque' in results_df.columns:
                st.markdown("### 🏷️ Analysis by Brand")
                brand_agg = results_df.groupby('marque')['avg_forecast'].agg(['sum', 'mean', 'count']).reset_index()
                brand_agg.columns = ['Brand', 'Total Forecast', 'Avg Forecast', 'Article Count']
                st.dataframe(brand_agg, use_container_width=True)
        else:
            st.info("👆 Run 'Forecast All' first to see summary statistics")
    
    with tab4:
        st.header("Data Preview")
        
        forecaster = st.session_state.forecaster
        
        # Raw data
        with st.expander("📄 Raw Data Sample", expanded=True):
            st.dataframe(forecaster.df_raw.head(100), use_container_width=True)
        
        # Clean data
        if forecaster.df_clean is not None:
            with st.expander("🧹 Clean Data Sample"):
                st.dataframe(forecaster.df_clean.head(100), use_container_width=True)
        
        # Grouped data
        if forecaster.grouped_data is not None:
            with st.expander("📊 Grouped Data Sample"):
                st.dataframe(forecaster.grouped_data.head(100), use_container_width=True)

else:
    # Welcome screen
    st.markdown("""
    <div class="info-box">
        <h2>👋 Welcome to the Sales Forecasting Dashboard!</h2>
        <p>This application helps you forecast sales using multiple advanced methods.</p>
        <h3>🚀 Getting Started:</h3>
        <ol>
            <li>Upload your sales data (CSV or Excel) using the sidebar</li>
            <li>Choose your forecasting frequency (yearly or monthly)</li>
            <li>Configure advanced settings if needed</li>
            <li>Click "Load/Reload Data" to initialize</li>
            <li>Start forecasting!</li>
        </ol>
        <h3>📋 Required Data Format:</h3>
        <ul>
            <li><strong>Ref Article:</strong> Article reference/ID</li>
            <li><strong>Année/Date:</strong> Year (for yearly) or Date (for monthly)</li>
            <li><strong>CA HT NET:</strong> Net sales amount</li>
            <li><strong>Optional:</strong> Designation, Marque, Famille, Sous Famille</li>
        </ul>
        <h3>🎯 Features:</h3>
        <ul>
            <li>Multiple forecasting methods (SMA, Exponential Smoothing, Linear Regression, ARIMA, Prophet, XGBoost)</li>
            <li>Detailed performance metrics (MAE, MSE, RMSE, MAPE, R²)</li>
            <li>Interactive visualizations</li>
            <li>Ensemble forecasting</li>
            <li>Cache management for faster processing</li>
            <li>Export results to CSV</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

