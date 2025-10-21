import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import ast
from datetime import datetime

# Import your forecaster - UPDATED to use IntegratedForecaster
from SalesAndQuantityForecaster import IntegratedForecaster

# Page configuration
st.set_page_config(
    page_title="Sales & Quantity Forecasting Dashboard",
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
    .qty-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
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
        
        # Initialize forecaster with DUAL forecasting support
        forecaster = IntegratedForecaster(
            dataframe=df,
            cache_dir="cache",
            ref_col="Ref Article",
            date_col=date_col,
            sales_col="CA HT NET",
            quantity_col="Qté Vendu",
            frequency=frequency
        )
        
        return forecaster
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def parse_json_field(value):
    """Parse JSON string or Python string representation"""
    if value is None or pd.isna(value):
        return None
    
    if isinstance(value, (list, dict)):
        return value
    
    if isinstance(value, str):
        try:
            return json.loads(value)
        except:
            pass
        try:
            return ast.literal_eval(value)
        except:
            pass
    
    return None

def parse_metrics(metrics_str):
    """Parse metrics from JSON string"""
    return parse_json_field(metrics_str)

def create_dual_forecast_chart(historical_periods, sales_values, qty_values, 
                                sales_forecast, qty_forecast, next_period, frequency):
    """Create dual-axis chart showing both sales and quantity forecasts"""
    
    # Parse values if they're JSON strings
    if isinstance(sales_values, str):
        sales_values = parse_json_field(sales_values)
    if isinstance(qty_values, str):
        qty_values = parse_json_field(qty_values)
    if isinstance(historical_periods, str):
        historical_periods = parse_json_field(historical_periods)
    
    # Validate data
    if sales_values is None or qty_values is None or historical_periods is None:
        return None
    
    # Convert to lists
    if isinstance(sales_values, np.ndarray):
        sales_values = sales_values.tolist()
    if isinstance(qty_values, np.ndarray):
        qty_values = qty_values.tolist()
    if isinstance(historical_periods, np.ndarray):
        historical_periods = historical_periods.tolist()
    
    # Period labels
    if frequency == 'monthly':
        period_labels = [str(p) for p in historical_periods]
        next_period_label = str(next_period)
    else:
        period_labels = [str(int(p)) for p in historical_periods]
        next_period_label = str(int(next_period))
    
    # Numeric x-axis
    x_numeric = list(range(len(historical_periods)))
    x_numeric_next = len(historical_periods)
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Sales traces (primary y-axis)
    fig.add_trace(go.Scatter(
        x=x_numeric,
        y=sales_values,
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        text=period_labels,
        hovertemplate='<b>%{text}</b><br>Sales: %{y:.2f}<extra></extra>',
        yaxis='y1'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_numeric[-1], x_numeric_next],
        y=[sales_values[-1], sales_forecast],
        mode='lines+markers',
        name='Sales Forecast',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=10, symbol='star'),
        text=[period_labels[-1], next_period_label],
        hovertemplate='<b>%{text}</b><br>Sales: %{y:.2f}<extra></extra>',
        yaxis='y1'
    ))
    
    # Quantity traces (secondary y-axis)
    fig.add_trace(go.Scatter(
        x=x_numeric,
        y=qty_values,
        mode='lines+markers',
        name='Historical Quantity',
        line=dict(color='#2ca02c', width=3),
        marker=dict(size=8, symbol='diamond'),
        text=period_labels,
        hovertemplate='<b>%{text}</b><br>Quantity: %{y:.0f}<extra></extra>',
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_numeric[-1], x_numeric_next],
        y=[qty_values[-1], qty_forecast],
        mode='lines+markers',
        name='Quantity Forecast',
        line=dict(color='#d62728', width=3, dash='dash'),
        marker=dict(size=10, symbol='star'),
        text=[period_labels[-1], next_period_label],
        hovertemplate='<b>%{text}</b><br>Quantity: %{y:.0f}<extra></extra>',
        yaxis='y2'
    ))
    
    # Update layout with dual y-axes
    fig.update_layout(
        title=f"Sales & Quantity Forecast ({frequency.capitalize()})",
        xaxis=dict(
            title="Period",
            tickmode='linear',
            tick0=0,
            dtick=1,
            ticktext=period_labels + [next_period_label],
            tickvals=x_numeric + [x_numeric_next]
        ),
        yaxis=dict(
            title=dict(text="Sales (CA HT NET)", font=dict(color="#1f77b4")),
            tickfont=dict(color="#1f77b4")
        ),
        yaxis2=dict(
            title=dict(text="Quantity (Qté Vendu)", font=dict(color="#2ca02c")),
            tickfont=dict(color="#2ca02c"),
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_forecast_chart(historical_periods, historical_values, forecast_value, 
                         next_period, frequency, value_type="Sales"):
    """Create single forecast visualization"""
    
    if isinstance(historical_values, str):
        historical_values = parse_json_field(historical_values)
    if isinstance(historical_periods, str):
        historical_periods = parse_json_field(historical_periods)
    
    if historical_values is None or historical_periods is None:
        return None
    
    if isinstance(historical_values, np.ndarray):
        historical_values = historical_values.tolist()
    if isinstance(historical_periods, np.ndarray):
        historical_periods = historical_periods.tolist()
    
    if frequency == 'monthly':
        period_labels = [str(p) for p in historical_periods]
        next_period_label = str(next_period)
    else:
        period_labels = [str(int(p)) for p in historical_periods]
        next_period_label = str(int(next_period))
    
    x_numeric = list(range(len(historical_periods)))
    x_numeric_next = len(historical_periods)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x_numeric,
        y=historical_values,
        mode='lines+markers',
        name=f'Historical {value_type}',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8),
        text=period_labels,
        hovertemplate=f'<b>%{{text}}</b><br>{value_type}: %{{y:.2f}}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[x_numeric[-1], x_numeric_next],
        y=[historical_values[-1], forecast_value],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='#ff7f0e', width=3, dash='dash'),
        marker=dict(size=10, symbol='star'),
        text=[period_labels[-1], next_period_label],
        hovertemplate=f'<b>%{{text}}</b><br>{value_type}: %{{y:.2f}}<extra></extra>'
    ))
    
    fig.update_xaxes(
        tickmode='linear',
        tick0=0,
        dtick=1,
        ticktext=period_labels + [next_period_label],
        tickvals=x_numeric + [x_numeric_next]
    )
    
    fig.update_layout(
        title=f"{value_type} Forecast ({frequency.capitalize()})",
        xaxis_title="Period",
        yaxis_title=value_type,
        hovermode='x unified',
        template='plotly_white',
        height=400
    )
    
    return fig

def create_metrics_chart(metrics_dict, method_name):
    """Create bar chart for metrics"""
    if not metrics_dict or all(pd.isna(v) for v in metrics_dict.values()):
        return None
    
    metrics_df = pd.DataFrame({
        'Metric': list(metrics_dict.keys()),
        'Value': list(metrics_dict.values())
    })
    
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

def display_method_comparison(result, value_type="sales"):
    """Display comparison of forecasting methods for sales or quantity"""
    methods = {
        'Simple Moving Average': f'{value_type}_sma_forecast',
        'Exponential Smoothing': f'{value_type}_es_forecast',
        'Linear Regression': f'{value_type}_lr_forecast',
        'ARIMA': f'{value_type}_arima_forecast',
        'Prophet': f'{value_type}_prophet_forecast',
        'XGBoost/Random Forest': f'{value_type}_xgb_forecast'
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
        title_prefix = "Sales" if value_type == "sales" else "Quantity"
        fig = px.bar(
            df_comp,
            x='Method',
            y='Forecast',
            title=f'{title_prefix} Forecast Comparison by Method',
            color='Forecast',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, template='plotly_white')
        avg_key = f'{value_type}_avg_forecast'
        if avg_key in result:
            fig.add_hline(
                y=result[avg_key],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Ensemble Average: {result[avg_key]:.2f}"
            )
        st.plotly_chart(fig, use_container_width=True)

# Main App
st.markdown('<p class="main-header">📊 Sales & Quantity Forecasting Dashboard</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_file = st.file_uploader(
        "Upload Sales Data",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your sales data in CSV or Excel format"
    )
    
    st.markdown("---")
    
    frequency = st.radio(
        "📅 Forecasting Frequency",
        options=['yearly', 'monthly'],
        index=0,
        help="Choose between yearly or monthly forecasting"
    )
    
    if frequency == 'yearly':
        date_col = st.text_input("Year Column Name", value="Année", help="Column containing year data")
    else:
        date_col = st.text_input("Date Column Name", value="Date", help="Column containing date data")
    
    st.markdown("---")
    
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
        force_recompute = st.checkbox("Force Recompute", value=False, help="Ignore cached results")
    
    st.markdown("---")
    
    with st.expander("🗑️ Cache Management"):
        st.write(f"**Current Frequency:** {frequency}")
        cache_path = Path("cache") / frequency
        if cache_path.exists():
            forecast_files = list((cache_path / "forecasts").glob("*.csv")) if (cache_path / "forecasts").exists() else []
            st.write(f"**Cache files:** {len(forecast_files)}")
        else:
            st.write("**Cache directory not found**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Clear Forecasts"):
                if st.session_state.forecaster:
                    st.session_state.forecaster.clear_forecast_cache()
                    st.success("Forecast cache cleared!")
        with col2:
            if st.button("Clear Summary"):
                if st.session_state.forecaster:
                    st.session_state.forecaster.clear_summary_cache()
                    st.success("Summary cache cleared!")
        with col3:
            if st.button("Clear All"):
                if st.session_state.forecaster:
                    st.session_state.forecaster.clear_all_caches()
                    st.success("All caches cleared!")

# Main content
if uploaded_file is not None:
    if not st.session_state.data_loaded or st.button("🔄 Load/Reload Data"):
        with st.spinner("Loading data..."):
            forecaster = load_data(uploaded_file, frequency, date_col)
            if forecaster:
                st.session_state.forecaster = forecaster
                st.session_state.data_loaded = True
                st.markdown('<div class="success-box">✅ Data loaded successfully!</div>', unsafe_allow_html=True)
                
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

if st.session_state.data_loaded and st.session_state.forecaster:
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast All", "🔍 Single Article", "📊 Summary Statistics", "📋 Data Preview"])
    
    with tab1:
        st.header("Forecast All Articles (Sales & Quantity)")
        
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
        
        if st.session_state.forecast_results is not None:
            results_df = st.session_state.forecast_results
            
            st.markdown("### 📊 Forecast Results")
            
            # Dual metrics display
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💰 Sales Metrics")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Avg Sales Forecast", f"{results_df['sales_avg_forecast'].mean():.2f}")
                with subcol2:
                    st.metric("Total Sales Forecast", f"{results_df['sales_avg_forecast'].sum():.2f}")
            
            with col2:
                st.markdown("#### 📦 Quantity Metrics")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Avg Qty Forecast", f"{results_df['qty_avg_forecast'].mean():.2f}")
                with subcol2:
                    st.metric("Total Qty Forecast", f"{results_df['qty_avg_forecast'].sum():.0f}")
            
            # Filters
            st.markdown("#### 🔍 Filter Results")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if 'marque' in results_df.columns:
                    marques = ['All'] + sorted([str(m) for m in results_df['marque'].dropna().unique()])
                    selected_marque = st.selectbox("Brand (Marque)", marques)
                else:
                    selected_marque = 'All'
            
            with col2:
                if 'famille' in results_df.columns:
                    familles = ['All'] + sorted([str(f) for f in results_df['famille'].dropna().unique()])
                    selected_famille = st.selectbox("Family (Famille)", familles)
                else:
                    selected_famille = 'All'
            
            with col3:
                min_sales = st.number_input("Min Sales Forecast", value=0.0, step=100.0)
            
            with col4:
                min_qty = st.number_input("Min Qty Forecast", value=0.0, step=10.0)
            
            # Apply filters
            filtered_df = results_df.copy()
            if selected_marque != 'All':
                filtered_df = filtered_df[filtered_df['marque'] == selected_marque]
            if selected_famille != 'All':
                filtered_df = filtered_df[filtered_df['famille'] == selected_famille]
            filtered_df = filtered_df[
                (filtered_df['sales_avg_forecast'] >= min_sales) &
                (filtered_df['qty_avg_forecast'] >= min_qty)
            ]
            
            st.markdown(f"**Showing {len(filtered_df)} of {len(results_df)} articles**")
            
            display_columns = ['ref_article', 'designation', 'marque', 'famille', 'next_period',
                             'sales_avg_forecast', 'qty_avg_forecast', 
                             'sales_avg', 'qty_avg', 'sales_trend_pct', 'qty_trend_pct', 'data_points']
            available_cols = [col for col in display_columns if col in filtered_df.columns]
            
            st.dataframe(
                filtered_df[available_cols].head(100),
                use_container_width=True,
                height=400
            )
            
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"dual_forecast_results_{frequency}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.header("Single Article Forecast (Sales & Quantity)")
        
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
            
            # Dual forecast visualization
            st.markdown("### 📈 Dual Forecast Visualization")
            dual_chart = create_dual_forecast_chart(
                result['historical_periods'],
                result['historical_sales'],
                result['historical_quantities'],
                result['sales_avg_forecast'],
                result['qty_avg_forecast'],
                result['next_period'],
                frequency
            )
            if dual_chart:
                st.plotly_chart(dual_chart, use_container_width=True)
            
            # Sales and Quantity results side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 💰 Sales Forecast Results")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Next Period", result['next_period'])
                    st.metric("Sales Forecast", f"{result['sales_avg_forecast']:.2f}")
                with subcol2:
                    st.metric("Avg Historical", f"{result['sales_avg']:.2f}")
                    st.metric("Trend", f"{result['sales_trend_pct']:.2f}%")
                
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    st.metric("Max", f"{result['sales_max']:.2f}")
                with subcol2:
                    st.metric("Min", f"{result['sales_min']:.2f}")
                with subcol3:
                    st.metric("Std Dev", f"{result['sales_std']:.2f}")
                
                # Sales method comparison
                st.markdown("#### 🔄 Sales Methods Comparison")
                display_method_comparison(result, "sales")
            
            with col2:
                st.markdown("### 📦 Quantity Forecast Results")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Next Period", result['next_period'])
                    st.metric("Qty Forecast", f"{result['qty_avg_forecast']:.2f}")
                with subcol2:
                    st.metric("Avg Historical", f"{result['qty_avg']:.2f}")
                    st.metric("Trend", f"{result['qty_trend_pct']:.2f}%")
                
                subcol1, subcol2, subcol3 = st.columns(3)
                with subcol1:
                    st.metric("Max", f"{result['qty_max']:.2f}")
                with subcol2:
                    st.metric("Min", f"{result['qty_min']:.2f}")
                with subcol3:
                    st.metric("Std Dev", f"{result['qty_std']:.2f}")
                
                # Quantity method comparison
                st.markdown("#### 🔄 Quantity Methods Comparison")
                display_method_comparison(result, "qty")
            
            # Detailed method performance
            st.markdown("### 📊 Detailed Method Performance")
            
            perf_tab1, perf_tab2 = st.tabs(["💰 Sales Methods", "📦 Quantity Methods"])
            
            methods_info = {
                'Simple Moving Average': ('sma_forecast', 'sma_metrics'),
                'Exponential Smoothing': ('es_forecast', 'es_metrics'),
                'Linear Regression': ('lr_forecast', 'lr_metrics'),
                'ARIMA': ('arima_forecast', 'arima_metrics'),
                'Prophet': ('prophet_forecast', 'prophet_metrics'),
                'XGBoost/RF': ('xgb_forecast', 'xgb_metrics')
            }
            
            with perf_tab1:
                for method_name, (forecast_key, metrics_key) in methods_info.items():
                    full_key = f'sales_{forecast_key}'
                    full_metrics = f'sales_{metrics_key}'
                    
                    if full_key in result and not pd.isna(result[full_key]):
                        with st.expander(f"📍 {method_name}"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.metric("Forecast Value", f"{result[full_key]:.2f}")
                                
                                if full_metrics in result:
                                    metrics = parse_metrics(result[full_metrics])
                                    if metrics:
                                        st.markdown("**Performance Metrics:**")
                                        for metric, value in metrics.items():
                                            if not pd.isna(value):
                                                st.write(f"- **{metric}:** {value:.4f}")
                            
                            with col2:
                                if full_metrics in result:
                                    metrics = parse_metrics(result[full_metrics])
                                    if metrics:
                                        chart = create_metrics_chart(metrics, method_name)
                                        if chart:
                                            st.plotly_chart(chart, use_container_width=True)
            
            with perf_tab2:
                for method_name, (forecast_key, metrics_key) in methods_info.items():
                    full_key = f'qty_{forecast_key}'
                    full_metrics = f'qty_{metrics_key}'
                    
                    if full_key in result and not pd.isna(result[full_key]):
                        with st.expander(f"📍 {method_name}"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.metric("Forecast Value", f"{result[full_key]:.2f}")
                                
                                if full_metrics in result:
                                    metrics = parse_metrics(result[full_metrics])
                                    if metrics:
                                        st.markdown("**Performance Metrics:**")
                                        for metric, value in metrics.items():
                                            if not pd.isna(value):
                                                st.write(f"- **{metric}:** {value:.4f}")
                            
                            with col2:
                                if full_metrics in result:
                                    metrics = parse_metrics(result[full_metrics])
                                    if metrics:
                                        chart = create_metrics_chart(metrics, method_name)
                                        if chart:
                                            st.plotly_chart(chart, use_container_width=True)
    
    with tab3:
        st.header("📊 Summary Statistics")
        
        forecaster = st.session_state.forecaster
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("📈 Generate Summary", type="primary", use_container_width=True):
                with st.spinner("Generating summary..."):
                    summary_df = forecaster.generate_summary(force_recompute=force_recompute)
                    st.session_state.summary_results = summary_df
                    st.success("✅ Summary generated!")
        
        # Check if we have summary results
        if 'summary_results' in st.session_state and st.session_state.summary_results is not None:
            summary_df = st.session_state.summary_results
        elif st.session_state.forecast_results is not None:
            summary_df = st.session_state.forecast_results
        else:
            summary_df = None
        
        if summary_df is not None and not summary_df.empty:
            st.markdown("### 📈 Overall Statistics")
            
            # Overall metrics in two rows
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Articles", len(summary_df))
            with col2:
                st.metric("Total Sales Forecast", f"{summary_df['sales_avg_forecast'].sum():.2f}")
            with col3:
                st.metric("Total Qty Forecast", f"{summary_df['qty_avg_forecast'].sum():.0f}")
            with col4:
                st.metric("Avg Data Points", f"{summary_df['data_points'].mean():.1f}")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Avg Sales per Article", f"{summary_df['sales_avg_forecast'].mean():.2f}")
            with col2:
                st.metric("Avg Qty per Article", f"{summary_df['qty_avg_forecast'].mean():.2f}")
            with col3:
                st.metric("Max Sales Forecast", f"{summary_df['sales_avg_forecast'].max():.2f}")
            with col4:
                st.metric("Max Qty Forecast", f"{summary_df['qty_avg_forecast'].max():.0f}")
            
            st.markdown("---")
            
            # Visualization tabs
            viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
                "📊 Top Articles", 
                "📈 Distribution Analysis", 
                "🏷️ Category Analysis", 
                "📉 Trend Analysis"
            ])
            
            with viz_tab1:
                st.markdown("### 🏆 Top Articles by Forecast")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 💰 Top 10 by Sales")
                    top_sales = summary_df.nlargest(10, 'sales_avg_forecast')[
                        ['ref_article', 'designation', 'sales_avg_forecast', 'sales_trend_pct']
                    ]
                    
                    fig = px.bar(
                        top_sales,
                        x='sales_avg_forecast',
                        y='ref_article',
                        orientation='h',
                        title='Top 10 Articles by Sales Forecast',
                        labels={'sales_avg_forecast': 'Sales Forecast', 'ref_article': 'Article'},
                        color='sales_trend_pct',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(top_sales, use_container_width=True, height=250)
                
                with col2:
                    st.markdown("#### 📦 Top 10 by Quantity")
                    top_qty = summary_df.nlargest(10, 'qty_avg_forecast')[
                        ['ref_article', 'designation', 'qty_avg_forecast', 'qty_trend_pct']
                    ]
                    
                    fig = px.bar(
                        top_qty,
                        x='qty_avg_forecast',
                        y='ref_article',
                        orientation='h',
                        title='Top 10 Articles by Quantity Forecast',
                        labels={'qty_avg_forecast': 'Quantity Forecast', 'ref_article': 'Article'},
                        color='qty_trend_pct',
                        color_continuous_scale='RdYlGn'
                    )
                    fig.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(top_qty, use_container_width=True, height=250)
            
            with viz_tab2:
                st.markdown("### 📊 Distribution Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 💰 Sales Forecast Distribution")
                    fig = px.histogram(
                        summary_df,
                        x='sales_avg_forecast',
                        nbins=50,
                        title='Sales Forecast Distribution',
                        labels={'sales_avg_forecast': 'Sales Forecast'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot
                    fig = px.box(
                        summary_df,
                        y='sales_avg_forecast',
                        title='Sales Forecast Box Plot',
                        labels={'sales_avg_forecast': 'Sales Forecast'}
                    )
                    fig.update_layout(height=300, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📦 Quantity Forecast Distribution")
                    fig = px.histogram(
                        summary_df,
                        x='qty_avg_forecast',
                        nbins=50,
                        title='Quantity Forecast Distribution',
                        labels={'qty_avg_forecast': 'Quantity Forecast'},
                        color_discrete_sequence=['#2ca02c']
                    )
                    fig.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot
                    fig = px.box(
                        summary_df,
                        y='qty_avg_forecast',
                        title='Quantity Forecast Box Plot',
                        labels={'qty_avg_forecast': 'Quantity Forecast'}
                    )
                    fig.update_layout(height=300, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Scatter plot: Sales vs Quantity
                st.markdown("#### 🔗 Sales vs Quantity Correlation")
                fig = px.scatter(
                    summary_df,
                    x='sales_avg_forecast',
                    y='qty_avg_forecast',
                    title='Sales vs Quantity Forecast',
                    labels={
                        'sales_avg_forecast': 'Sales Forecast',
                        'qty_avg_forecast': 'Quantity Forecast'
                    },
                    hover_data=['ref_article', 'designation'],
                    trendline='ols'
                )
                fig.update_layout(height=500, template='plotly_white')
                st.plotly_chart(fig, use_container_width=True)
            
            with viz_tab3:
                st.markdown("### 🏷️ Category Analysis")
                
                # By Brand (Marque)
                if 'marque' in summary_df.columns:
                    st.markdown("#### 📱 Analysis by Brand")
                    
                    brand_summary = summary_df.groupby('marque').agg({
                        'sales_avg_forecast': ['sum', 'mean', 'count'],
                        'qty_avg_forecast': ['sum', 'mean']
                    }).reset_index()
                    brand_summary.columns = ['Brand', 'Total Sales', 'Avg Sales', 'Count', 'Total Qty', 'Avg Qty']
                    brand_summary = brand_summary.sort_values('Total Sales', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.bar(
                            brand_summary.head(15),
                            x='Brand',
                            y='Total Sales',
                            title='Total Sales Forecast by Brand (Top 15)',
                            color='Total Sales',
                            color_continuous_scale='Blues'
                        )
                        fig.update_layout(height=400, template='plotly_white', xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.bar(
                            brand_summary.head(15),
                            x='Brand',
                            y='Total Qty',
                            title='Total Quantity Forecast by Brand (Top 15)',
                            color='Total Qty',
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=400, template='plotly_white', xaxis_tickangle=-45)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(brand_summary.head(20), use_container_width=True, height=300)
                
                # By Family (Famille)
                if 'famille' in summary_df.columns:
                    st.markdown("#### 📂 Analysis by Family")
                    
                    family_summary = summary_df.groupby('famille').agg({
                        'sales_avg_forecast': ['sum', 'mean', 'count'],
                        'qty_avg_forecast': ['sum', 'mean']
                    }).reset_index()
                    family_summary.columns = ['Family', 'Total Sales', 'Avg Sales', 'Count', 'Total Qty', 'Avg Qty']
                    family_summary = family_summary.sort_values('Total Sales', ascending=False)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig = px.pie(
                            family_summary.head(10),
                            values='Total Sales',
                            names='Family',
                            title='Sales Distribution by Family (Top 10)'
                        )
                        fig.update_layout(height=400, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        fig = px.pie(
                            family_summary.head(10),
                            values='Total Qty',
                            names='Family',
                            title='Quantity Distribution by Family (Top 10)'
                        )
                        fig.update_layout(height=400, template='plotly_white')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    st.dataframe(family_summary.head(20), use_container_width=True, height=300)
            
            with viz_tab4:
                st.markdown("### 📉 Trend Analysis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 💰 Sales Trend Distribution")
                    
                    # Categorize trends
                    summary_df['sales_trend_category'] = pd.cut(
                        summary_df['sales_trend_pct'],
                        bins=[-float('inf'), -10, -5, 5, 10, float('inf')],
                        labels=['Strong Decline', 'Decline', 'Stable', 'Growth', 'Strong Growth']
                    )
                    
                    trend_counts = summary_df['sales_trend_category'].value_counts()
                    
                    fig = px.pie(
                        values=trend_counts.values,
                        names=trend_counts.index,
                        title='Sales Trend Categories',
                        color_discrete_sequence=['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60']
                    )
                    fig.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogram
                    fig = px.histogram(
                        summary_df,
                        x='sales_trend_pct',
                        nbins=50,
                        title='Sales Trend % Distribution',
                        labels={'sales_trend_pct': 'Trend %'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    fig.update_layout(height=300, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📦 Quantity Trend Distribution")
                    
                    # Categorize trends
                    summary_df['qty_trend_category'] = pd.cut(
                        summary_df['qty_trend_pct'],
                        bins=[-float('inf'), -10, -5, 5, 10, float('inf')],
                        labels=['Strong Decline', 'Decline', 'Stable', 'Growth', 'Strong Growth']
                    )
                    
                    trend_counts = summary_df['qty_trend_category'].value_counts()
                    
                    fig = px.pie(
                        values=trend_counts.values,
                        names=trend_counts.index,
                        title='Quantity Trend Categories',
                        color_discrete_sequence=['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91cf60']
                    )
                    fig.update_layout(height=400, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogram
                    fig = px.histogram(
                        summary_df,
                        x='qty_trend_pct',
                        nbins=50,
                        title='Quantity Trend % Distribution',
                        labels={'qty_trend_pct': 'Trend %'},
                        color_discrete_sequence=['#2ca02c']
                    )
                    fig.update_layout(height=300, template='plotly_white')
                    st.plotly_chart(fig, use_container_width=True)
                
                # Articles with highest growth
                st.markdown("#### 🚀 Fastest Growing Articles")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**By Sales Growth**")
                    top_growth_sales = summary_df.nlargest(10, 'sales_trend_pct')[
                        ['ref_article', 'designation', 'sales_avg_forecast', 'sales_trend_pct']
                    ]
                    st.dataframe(top_growth_sales, use_container_width=True, height=300)
                
                with col2:
                    st.markdown("**By Quantity Growth**")
                    top_growth_qty = summary_df.nlargest(10, 'qty_trend_pct')[
                        ['ref_article', 'designation', 'qty_avg_forecast', 'qty_trend_pct']
                    ]
                    st.dataframe(top_growth_qty, use_container_width=True, height=300)
        else:
            st.info("📊 No summary data available. Please run forecasts first or generate summary.")
    
    with tab4:
        st.header("📋 Data Preview")
        
        forecaster = st.session_state.forecaster
        
        preview_option = st.radio(
            "Select Data to Preview",
            options=["Raw Data", "Cleaned Data", "Grouped Data"],
            horizontal=True
        )
        
        if preview_option == "Raw Data":
            st.markdown("### 📄 Raw Data")
            if forecaster.df_raw is not None:
                st.markdown(f"**Total Rows:** {len(forecaster.df_raw)}")
                st.markdown(f"**Columns:** {', '.join(forecaster.df_raw.columns.tolist())}")
                st.dataframe(forecaster.df_raw.head(100), use_container_width=True, height=500)
                
                # Download option
                csv = forecaster.df_raw.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Raw Data",
                    data=csv,
                    file_name=f"raw_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        elif preview_option == "Cleaned Data":
            st.markdown("### 🧹 Cleaned Data")
            if forecaster.df_clean is None:
                forecaster.clean_data()
            
            if forecaster.df_clean is not None:
                st.markdown(f"**Total Rows:** {len(forecaster.df_clean)}")
                st.markdown(f"**Columns:** {', '.join(forecaster.df_clean.columns.tolist())}")
                st.dataframe(forecaster.df_clean.head(100), use_container_width=True, height=500)
                
                # Download option
                csv = forecaster.df_clean.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Cleaned Data",
                    data=csv,
                    file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        else:  # Grouped Data
            st.markdown("### 📊 Grouped Data")
            if forecaster.grouped_data is None:
                forecaster.prepare_data()
            
            if forecaster.grouped_data is not None:
                st.markdown(f"**Total Rows:** {len(forecaster.grouped_data)}")
                st.markdown(f"**Unique Articles:** {forecaster.grouped_data['Ref Article'].nunique()}")
                st.markdown(f"**Columns:** {', '.join(forecaster.grouped_data.columns.tolist())}")
                st.dataframe(forecaster.grouped_data.head(100), use_container_width=True, height=500)
                
                # Download option
                csv = forecaster.grouped_data.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Grouped Data",
                    data=csv,
                    file_name=f"grouped_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem;">
        <h2>👋 Welcome to the Sales & Quantity Forecasting Dashboard</h2>
        <p style="font-size: 1.2rem; color: #666;">
            Upload your sales data to get started with advanced dual forecasting
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ### 📋 How to Use:
        
        1. **Upload Data**: Use the sidebar to upload your CSV or Excel file
        2. **Configure**: Select frequency (yearly/monthly) and advanced settings
        3. **Forecast**: Run forecasts for all articles or individual items
        4. **Analyze**: Explore comprehensive statistics and visualizations
        5. **Export**: Download results for further analysis
        
        ### ✨ Features:
        
        - 💰 **Dual Forecasting**: Simultaneous Sales & Quantity predictions
        - 📊 **6 Methods**: SMA, Exponential Smoothing, Linear Regression, ARIMA, Prophet, XGBoost
        - 📈 **Detailed Metrics**: MAE, MSE, RMSE, MAPE, R² for each method
        - 🎯 **Smart Caching**: Fast performance with intelligent result caching
        - 📉 **Rich Visualizations**: Interactive charts and comprehensive analytics
        - 🔍 **Flexible Filtering**: By brand, family, forecast values, and more
        
        ### 📊 Required Data Columns:
        
        - `Ref Article`: Article reference/ID
        - `CA HT NET`: Sales amount
        - `Qté Vendu`: Quantity sold
        - `Année` or `Date`: Time period (depending on frequency)
        - Optional: `Marque`, `Famille`, `Designation` for categorization
        """)

# Footer
