import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from sales_forecaster import SalesForecaster  # your class file

# -------------------------------------------
# 🎨 Streamlit UI Configuration
# -------------------------------------------
st.set_page_config(page_title="Sales Forecasting App", layout="wide")

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
    # Handle both CSV and Excel formats
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        st.error("❌ Unsupported file format. Please upload a CSV or XLSX file.")
        st.stop()

    st.success(f"✅ Data loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")

    # Show raw data preview
    with st.expander("🔍 Preview raw data"):
        st.dataframe(df.head(20))

    # Initialize forecaster
    forecaster = SalesForecaster(dataframe=df)

    # -------------------------------------------
    # ⚙️ Sidebar Configuration
    # -------------------------------------------
    st.sidebar.header("⚙️ Forecast Settings")
    period = st.sidebar.slider("SMA Period", min_value=2, max_value=5, value=3)
    alpha = st.sidebar.slider("Exponential Smoothing Alpha", min_value=0.1, max_value=0.9, value=0.3, step=0.1)

    # -------------------------------------------
    # 🚀 User Actions
    # -------------------------------------------
    option = st.radio(
        "Select action:", 
        ["Forecast All Articles", "Forecast Single Article", "Summary Report"]
    )

    if option == "Forecast All Articles":
        with st.spinner("Running forecasts for all articles..."):
            forecast_df = forecaster.forecast_all_articles(period=period, alpha=alpha)
        
        st.subheader("📊 Forecast Results")
        st.dataframe(forecast_df.head(20))

        total_forecast = forecast_df["avg_forecast"].sum()
        st.metric(label="💰 Total Forecasted Sales (DT)", value=f"{total_forecast:,.2f}")

        # Download button
        csv_buffer = BytesIO()
        forecast_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Forecast Results as CSV",
            data=csv_buffer.getvalue(),
            file_name="forecast_results.csv",
            mime="text/csv"
        )

    elif option == "Forecast Single Article":
        forecaster.prepare_data()
        articles = sorted(forecaster.grouped_data["Ref Article"].unique().tolist())
        selected_article = st.selectbox("Select an Article Reference", articles)

        if st.button("🔮 Generate Forecast"):
            st.info(f"Forecasting for Article: {selected_article}")
            result = forecaster.forecast_article(selected_article, period, alpha)

            if result:
                st.write(f"**Designation:** {result['designation']}")
                st.write(f"**Brand:** {result['marque']} | **Family:** {result['famille']}")
                st.write(f"**Trend:** {result['trend_pct']:+.2f}% | **Next Year Forecast:** {result['avg_forecast']:.2f} DT")

                # Plot data
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                # Historical + Forecast
                years = result['historical_years']
                values = result['historical_values']
                axes[0].plot(years, values, 'o-', label='Historical', color='#3b82f6')
                axes[0].plot([result['next_year']], [result['avg_forecast']], 'o', label='Forecast', color='#10b981')
                axes[0].legend()
                axes[0].set_title(f"{selected_article} - {result['designation']}")

                # Comparison
                methods = ['SMA', 'Exp. Smoothing', 'Linear Reg.', 'Average']
                forecasts = [
                    result['sma_forecast'], result['es_forecast'], 
                    result['lr_forecast'], result['avg_forecast']
                ]
                axes[1].bar(methods, forecasts, color=['#3b82f6', '#8b5cf6', '#f59e0b', '#10b981'])
                axes[1].set_title("Forecast Method Comparison")

                st.pyplot(fig)

    elif option == "Summary Report":
        with st.spinner("Generating summary..."):
            forecaster.forecast_all_articles(period=period, alpha=alpha)
            st.subheader("📋 Summary Report")
            forecaster.summary_report()

else:
    st.info("👆 Please upload a `.csv` or `.xlsx` file to begin forecasting.")
