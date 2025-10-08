# app_streamlit.py
import streamlit as st
import pandas as pd
from io import BytesIO
from sales_forecaster import SalesForecaster
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Sales Forecaster", layout="wide")

st.title("📈 Sales Forecasting Dashboard (Cached & Fast Mode)")

# ------------- Upload / Load -------------
uploaded_file = st.file_uploader("Upload CSV or XLSX ", type=["csv", "xlsx"])
if uploaded_file is None:
    st.info("Please upload a dataset to begin (CSV/XLSX).")
    st.stop()

# Read file
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.success(f"Loaded {uploaded_file.name} — {df.shape[0]} rows.")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20))

# Initialize forecaster
forecaster = SalesForecaster(dataframe=df, cache_dir="cache")

# Clean & prepare (quick)
with st.spinner("Cleaning & preparing data..."):
    forecaster.clean_data()
    forecaster.prepare_data()
st.success("Data ready.")

# ---------------- Sidebar: settings ----------------
st.sidebar.header("Settings")
period = st.sidebar.slider("SMA period", 2, 5, 3)
alpha = st.sidebar.slider("Exp Smoothing alpha", 0.1, 0.9, 0.3, 0.1)
fast_mode = st.sidebar.checkbox("Enable fast mode (skip heavy models for short series)", value=True)
force_recompute = st.sidebar.checkbox("Force recompute (bypass caches)", value=False)

st.sidebar.markdown("**Models to compute / include in ensemble**")
models_selected = st.sidebar.multiselect(
    "Select methods",
    options=["SMA", "ExpSmoothing", "LinearReg", "ARIMA", "PROPHET", "XGBOOST"],
    default=["SMA", "ExpSmoothing", "LinearReg", "XGBOOST"]
)

st.sidebar.markdown("---")
if st.sidebar.button("🔁 Refresh all caches (clear)"):
    forecaster.clear_all_caches()
    st.sidebar.success("Caches cleared. Forecasts and summary will recompute when requested.")

# ---------------- Main tabs ----------------
tab1, tab2, tab3 = st.tabs(["📊 Forecast All", "🔎 Single Article", "📋 Summary & Reports"])

# ---------------- TAB 1 Forecast All ----------------
with tab1:
    st.header("Forecast All Articles (cached)")
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔮 Run / Refresh Forecasts for All"):
            with st.spinner("Forecasting... this may take some time for many articles (uses cache)..."):
                progress_bar = st.progress(0)
                total_articles = len(forecaster.grouped_data[forecaster.ref_col].unique())
                def prog(i, tot):
                    progress_bar.progress(int(i / tot * 100))
                df_forecasts = forecaster.forecast_all_articles(period=period, alpha=alpha,
                                                               force_recompute=force_recompute,
                                                               include_methods=models_selected,
                                                               fast_mode=fast_mode,
                                                               progress_callback=prog)
                st.success("Forecasts computed and cached.")
    with col2:
        st.info("Use 'Run / Refresh' to (re)compute forecasts. Cached per-article forecasts are used for fast reloads.")

    # show top 20
    summary_df = forecaster.generate_summary(force_recompute=False)
    if summary_df is None or summary_df.empty:
        st.warning("No forecasts available — run forecasting first.")
    else:
        st.subheader("Top Forecasts")
        st.dataframe(summary_df.head(20), use_container_width=True)
        # download
        csv = summary_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Summary CSV", data=csv, file_name="summary_forecasts.csv", mime="text/csv")

# ---------------- TAB 2 Single Article ----------------
with tab2:
    st.header("Single Article Analysis (fast & cached)")
    articles = sorted(forecaster.grouped_data[forecaster.ref_col].unique().tolist())
    selected_article = st.selectbox("Select an article", articles)
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        if st.button("🔍 Show Forecast for selected article"):
            with st.spinner("Loading forecast (uses cache when possible)..."):
                res = forecaster.forecast_article(
                    selected_article,
                    period=period,
                    alpha=alpha,
                    force_recompute=force_recompute,
                    include_methods=models_selected,
                    fast_mode=fast_mode
                )

                if res is None:
                    st.error("No historical data for this article.")
                else:
                    st.markdown(f"### {res.get('designation') or selected_article}")
                    st.write(f"Brand: {res.get('marque')} | Family: {res.get('famille')}")
                    st.write(f"Data points: {res.get('data_points')}")
                    st.write(f"Trend % (historical): {res.get('trend_pct'):+.2f}%")
                    st.metric("Next year (ensemble)", f"{res.get('avg_forecast'):.2f} DT")

                    # ✅ Fix: safely convert to lists if strings
                    years = res.get('historical_years', [])
                    values = res.get('historical_values', [])
                    next_y = res.get('next_year')

                    if isinstance(years, str):
                        try:
                            years = ast.literal_eval(years)
                        except Exception:
                            years = [years]

                    if isinstance(values, str):
                        try:
                            values = ast.literal_eval(values)
                        except Exception:
                            values = [values]

                    # Ensure same length
                    if len(years) != len(values):
                        min_len = min(len(years), len(values))
                        years = years[:min_len]
                        values = values[:min_len]

                    # interactive historical + forecast chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=years, y=values, mode='lines+markers', name='Historical'
                    ))
                    fig.add_trace(go.Scatter(
                        x=[next_y],
                        y=[res['avg_forecast']],
                        mode='markers+text',
                        text=["Forecast"],
                        name='Ensemble Forecast'
                    ))
                    fig.update_layout(
                        title=f"{selected_article} - Historical vs Forecast",
                        xaxis_title="Year",
                        yaxis_title="CA HT NET (DT)"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # model comparison bars
                    methods, values_b = [], []
                    for m in [
                        "sma_forecast", "es_forecast", "lr_forecast",
                        "arima_forecast", "prophet_forecast", "xgb_forecast"
                    ]:
                        if m in res and not pd.isna(res[m]):
                            methods.append(m.replace("_forecast", "").upper())
                            values_b.append(res[m])

                    if methods:
                        fig2 = go.Figure([go.Bar(x=methods, y=values_b)])
                        fig2.update_layout(
                            title="Model comparison",
                            yaxis_title="Forecast (DT)"
                        )
                        st.plotly_chart(fig2, use_container_width=True)

    with btn_col2:
        if st.button("💾 Force recompute & update cache for this article"):
            with st.spinner("Recomputing and caching..."):
                res = forecaster.forecast_article(
                    selected_article,
                    period=period,
                    alpha=alpha,
                    force_recompute=True,
                    include_methods=models_selected,
                    fast_mode=fast_mode
                )
                st.success("Recomputed and cached.")
                st.experimental_rerun()
# ---------------- TAB 3 Summary & Reports ----------------
with tab3:
    st.header("Summary & Reports (instant via cache)")
    refresh = st.button("🔁 Recompute Summary (force)")
    if refresh:
        with st.spinner("Recomputing summary..."):
            df_summary = forecaster.generate_summary(force_recompute=True)
            st.success("Summary recomputed.")
    else:
        df_summary = forecaster.generate_summary(force_recompute=False)

    if df_summary is None or df_summary.empty:
        st.warning("No summary available. Run forecasts first.")
    else:
        st.subheader("Top 10 by forecast")
        st.dataframe(df_summary.head(10)[["ref_article", "designation", "marque", "avg_forecast", "trend_label", "trend_pct"]])

        st.subheader("Trend distribution")
        dist = df_summary['trend_label'].value_counts().reset_index()
        dist.columns = ['trend', 'count']
        fig = px.pie(dist, names='trend', values='count', title='Trend Distribution')
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Trend scatter (forecast vs trend_pct)")
        fig2 = px.scatter(df_summary, x='avg_forecast', y='trend_pct', hover_data=['ref_article', 'designation'])
        st.plotly_chart(fig2, use_container_width=True)
