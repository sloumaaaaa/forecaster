import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go
from sales_forecaster import SalesForecaster

# ------------------------------------------------------------
# 🎨 Streamlit UI Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 2rem;}
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Sales Forecasting Dashboard")
st.markdown("Use statistical models to predict **CA HT NET** per article with multiple forecasting techniques.")

# ------------------------------------------------------------
# 📤 File Upload Section
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload your sales dataset (.csv or .xlsx)", 
    type=["csv", "xlsx"],
    help="File must contain columns: Ref Article, Année, CA HT NET, Designation, Marque, Famille"
)

if uploaded_file is not None:
    # Read file
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        st.stop()

    st.success(f"✅ {uploaded_file.name} loaded successfully ({len(df)} rows).")

    with st.expander("👁 Preview raw data", expanded=False):
        st.dataframe(df.head(20))

    # ------------------------------------------------------------
    # ⚙️ Initialize Forecaster
    # ------------------------------------------------------------
    forecaster = SalesForecaster(dataframe=df)
    with st.spinner("🧹 Cleaning and preparing data..."):
        forecaster.clean_data()
        grouped_data = forecaster.prepare_data()
    st.success(f"✅ Data ready ({grouped_data['Ref Article'].nunique()} unique articles).")

    # Sidebar configuration
    st.sidebar.header("⚙️ Forecast Settings")
    period = st.sidebar.slider("SMA Period", 2, 5, 3)
    alpha = st.sidebar.slider("Exponential Smoothing Alpha", 0.1, 0.9, 0.3, 0.1)
    st.sidebar.caption("These affect SMA and exponential smoothing calculations.")

    # ------------------------------------------------------------
    # 🧭 Tabs
    # ------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs([
        "🔮 Forecast All Articles",
        "📊 Single Article Analysis",
        "📋 Summary & Reports"
    ])

    # ============================================================
    # TAB 1 – Forecast All Articles
    # ============================================================
    with tab1:
        st.header("🔮 Forecast All Articles")

        if st.button("🚀 Run Forecasts for All Articles"):
            with st.spinner("Running forecasts across all articles..."):
                forecast_df = forecaster.forecast_all_articles(period=period, alpha=alpha)

            # --- Key Metrics ---
            c1, c2, c3 = st.columns(3)
            c1.metric("📦 Articles Forecasted", len(forecast_df))
            c2.metric("💰 Total Forecast (DT)", f"{forecast_df['avg_forecast'].sum():,.2f}")
            c3.metric("📈 Avg Trend (%)", f"{forecast_df['trend_pct'].mean():+.2f}")

            # --- Data Table ---
            st.subheader("📊 Forecast Results")
            st.dataframe(forecast_df.head(20), use_container_width=True)

            # --- Download Button ---
            csv_buf = BytesIO()
            forecast_df.to_csv(csv_buf, index=False)
            st.download_button(
                label="💾 Download Forecast Results (CSV)",
                data=csv_buf.getvalue(),
                file_name="forecast_results.csv",
                mime="text/csv"
            )

    # ============================================================
    # TAB 2 – Forecast Single Article
    # ============================================================
    with tab2:
        st.header("📊 Analyze Forecast for a Single Article")
        forecaster.prepare_data()
        article_list = sorted(forecaster.grouped_data['Ref Article'].unique().tolist())
        selected_article = st.selectbox("Select Article Reference", article_list)

        if st.button("🔍 Generate Forecast for Selected Article"):
            with st.spinner(f"Forecasting for article {selected_article}..."):
                result = forecaster.forecast_article(selected_article, period=period, alpha=alpha)

            if result:
                # --- Info Summary ---
                st.markdown(f"### 🏷 {result['designation']}")
                st.write(f"**Brand:** {result['marque']} | **Family:** {result['famille']}")
                st.write(f"**Trend:** {result['trend_pct']:+.2f}% | **Next Year Forecast:** {result['avg_forecast']:.2f} DT")

                # --- Historical + Forecast Plot ---
                years = result['historical_years']
                values = result['historical_values']
                next_year = result['next_year']

                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=years, y=values, mode='lines+markers',
                                          name='Historical', line=dict(color='#3b82f6')))
                fig1.add_trace(go.Scatter(x=[next_year], y=[result['avg_forecast']],
                                          mode='markers+text', text=["Forecast"],
                                          name='Forecast', marker=dict(size=12, color='#10b981')))
                fig1.add_hline(y=result['avg_sales'], line_dash="dash", line_color="#ef4444",
                               annotation_text="Avg Sales")
                fig1.update_layout(title=f"{selected_article} – Forecast Overview",
                                   xaxis_title="Year", yaxis_title="CA HT NET (DT)")
                st.plotly_chart(fig1, use_container_width=True)

                # --- Comparison Bar Chart ---
                methods = ["SMA", "Exp. Smoothing", "Linear Reg.", "ARIMA", "Average"]
                forecasts = [
                    result["sma_forecast"],
                    result["es_forecast"],
                    result["lr_forecast"],
                    result["arima_forecast"],
                    result["avg_forecast"]
                ]
                colors = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ec4899', '#10b981']

                fig2 = go.Figure([go.Bar(x=methods, y=forecasts, marker_color=colors)])
                fig2.update_layout(title="Forecast Model Comparison", yaxis_title="Predicted Sales (DT)")
                st.plotly_chart(fig2, use_container_width=True)

    # ============================================================
    # TAB 3 – Summary Report
    # ============================================================
    with tab3:
        st.header("📋 Global Summary Report")
        with st.spinner("Compiling summary report..."):
            forecaster.forecast_all_articles(period=period, alpha=alpha)
            forecast_df = forecaster.forecast_results

        # --- Summary Metrics ---
        c1, c2, c3 = st.columns(3)
        c1.metric("📦 Articles Forecasted", len(forecast_df))
        c2.metric("💰 Total Forecast", f"{forecast_df['avg_forecast'].sum():,.2f} DT")
        c3.metric("📈 Positive Trends", f"{(forecast_df['trend_pct']>0).sum()} articles")

        st.subheader("🏆 Top 10 Articles by Forecast Value")
        st.dataframe(forecast_df.head(10)[
            ["ref_article", "designation", "marque", "avg_forecast", "trend_pct"]
        ])

        st.subheader("📉 Trend Distribution")
        st.bar_chart(forecast_df["trend_pct"])

else:
    st.info("👆 Upload a dataset to begin forecasting.")
