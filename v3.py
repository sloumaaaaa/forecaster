import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go
from sales_forecaster import SalesForecaster

# -------------------------------------------
# 🎨 Streamlit UI Configuration
# -------------------------------------------
st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

# Custom CSS for a clean look
st.markdown("""
    <style>
    .main {
        padding: 1.5rem;
        background-color: #f9fafb;
    }
    .stMetric {
        background-color: white;
        border-radius: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Sales Forecasting Dashboard")
st.markdown("Predict **CA HT NET** for each article using statistical models.")

# -------------------------------------------
# 📤 File Upload Section
# -------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload your sales data file (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    # Read CSV/Excel
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("❌ Unsupported file format.")
        st.stop()

    st.success(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    with st.expander("🔍 Preview raw data"):
        st.dataframe(df.head(20))

    # Initialize forecaster
    forecaster = SalesForecaster(dataframe=df)

    # -------------------------------------------
    # ⚙️ Sidebar Configuration
    # -------------------------------------------
    st.sidebar.header("⚙️ Forecast Settings")
    models_selected = st.sidebar.multiselect(
        "Select forecasting models:",
        options=["SMA", "Exponential Smoothing", "Linear Regression", "ARIMA"],
        default=["SMA", "Exponential Smoothing", "Linear Regression", "ARIMA"]
    )
    period = st.sidebar.slider("SMA Period", 2, 5, 3)
    alpha = st.sidebar.slider("Exponential Smoothing Alpha", 0.1, 0.9, 0.3, 0.1)

    theme = st.sidebar.radio("🎨 Theme:", ["Light", "Dark"])
    if theme == "Dark":
        st.markdown("<style>body {background-color:#1e1e1e; color:white;}</style>", unsafe_allow_html=True)

    # -------------------------------------------
    # 🧭 Tabs Layout
    # -------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Forecast All", "🔮 Single Article", "📋 Summary Report"])

    # ===========================================
    # TAB 1 – Forecast All Articles
    # ===========================================
    with tab1:
        with st.spinner("Running forecasts for all articles..."):
            forecast_df = forecaster.forecast_all_articles(period=period, alpha=alpha)

        # Top metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("📦 Articles Forecasted", len(forecast_df))
        col2.metric("💰 Total Forecast (DT)", f"{forecast_df['avg_forecast'].sum():,.2f}")
        col3.metric("📈 Avg Trend", f"{forecast_df['trend_pct'].mean():+.2f}%")

        # Data preview
        st.subheader("📊 Forecast Results")
        st.dataframe(forecast_df.head(20))

        # Download
        csv_buffer = BytesIO()
        forecast_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Forecast CSV",
            data=csv_buffer.getvalue(),
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    # ===========================================
    # TAB 2 – Forecast Single Article
    # ===========================================
    with tab2:
        forecaster.prepare_data()
        articles = sorted(forecaster.grouped_data["Ref Article"].unique().tolist())
        selected_article = st.selectbox("Select Article Reference", articles)

        if st.button("🔮 Generate Forecast"):
            st.info(f"Forecasting for: {selected_article}")
            result = forecaster.forecast_article(selected_article, period, alpha)

            if result:
                st.write(f"**Designation:** {result['designation']}")
                st.write(f"**Brand:** {result['marque']} | **Family:** {result['famille']}")
                st.write(f"**Trend:** {result['trend_pct']:+.2f}% | **Next Year Forecast:** {result['avg_forecast']:.2f} DT")

                # Plotly interactive chart
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
                fig1.update_layout(title=f"{selected_article} – {result['designation']}",
                                   xaxis_title="Year", yaxis_title="CA HT NET (DT)")
                st.plotly_chart(fig1, use_container_width=True)

                # Comparison chart
                methods, forecasts, colors = [], [], []
                if "SMA" in models_selected:
                    methods.append("SMA"); forecasts.append(result['sma_forecast']); colors.append('#3b82f6')
                if "Exponential Smoothing" in models_selected:
                    methods.append("Exp. Smoothing"); forecasts.append(result['es_forecast']); colors.append('#8b5cf6')
                if "Linear Regression" in models_selected:
                    methods.append("Linear Reg."); forecasts.append(result['lr_forecast']); colors.append('#f59e0b')
                if "ARIMA" in models_selected:
                    methods.append("ARIMA"); forecasts.append(result['arima_forecast']); colors.append('#ec4899')
                methods.append("Average"); forecasts.append(result['avg_forecast']); colors.append('#10b981')

                fig2 = go.Figure([go.Bar(x=methods, y=forecasts, marker_color=colors)])
                fig2.update_layout(title="Forecast Method Comparison",
                                   yaxis_title="Forecast CA HT NET (DT)")
                st.plotly_chart(fig2, use_container_width=True)

    # ===========================================
    # TAB 3 – Summary Report
    # ===========================================
    with tab3:
        with st.spinner("Generating summary..."):
            forecaster.forecast_all_articles(period=period, alpha=alpha)
            st.subheader("📋 Summary Report")

            forecast_df = forecaster.forecast_results
            total = forecast_df['avg_forecast'].sum()
            avg = forecast_df['avg_forecast'].mean()
            pos_trend = (forecast_df['trend_pct'] > 0).sum()
            neg_trend = (forecast_df['trend_pct'] < 0).sum()

            st.write(f"**Total Articles:** {len(forecast_df)}")
            st.write(f"**Forecast Year:** {int(forecast_df['next_year'].iloc[0])}")
            st.write(f"**Total Forecasted CA HT NET:** {total:,.2f} DT")
            st.write(f"**Average Forecast per Article:** {avg:,.2f} DT")
            st.write(f"**Positive Trend:** {pos_trend} articles ({pos_trend/len(forecast_df)*100:.1f}%)")
            st.write(f"**Negative Trend:** {neg_trend} articles ({neg_trend/len(forecast_df)*100:.1f}%)")

            st.markdown("### 🏆 Top 5 Articles by Forecast")
            st.dataframe(forecast_df.head(5)[
                ["ref_article", "designation", "marque", "avg_forecast", "trend_pct"]
            ])

else:
    st.info("👆 Please upload a `.csv` or `.xlsx` file to begin forecasting.")
