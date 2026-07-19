import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data.loader import load_forex_data
from src.features.engineer import load_config
from predict import load_artifacts, preprocess

# ── Streamlit Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Forex Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Forex Ensemble Prediction Dashboard")
st.markdown("Visualise the predictive performance of our ML models against historical and recent data.")

# ── Constants & Helpers ──────────────────────────────────────────────────────
DATA_PATH = os.environ.get("DATA_PATH", "Forex_Data.csv")
MODEL_DIR = os.environ.get("MODEL_DIR", "outputs")
CONFIG_PATH = os.environ.get("FEAT_CONFIG", "config/features.yaml")

@st.cache_data
def get_config():
    return load_config(CONFIG_PATH)

@st.cache_data
def get_raw_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found at {DATA_PATH}")
        st.stop()
    return load_forex_data(DATA_PATH)

@st.cache_resource
def get_model_artifacts(model_name):
    if not os.path.exists(MODEL_DIR):
        st.error(f"Model directory not found at {MODEL_DIR}")
        st.stop()
    try:
        return load_artifacts(MODEL_DIR, model_name)
    except Exception as e:
        st.error(f"Failed to load artifacts for {model_name}: {e}")
        st.stop()

@st.cache_data
def get_predictions(model_name, _df_raw, _feat_cfg):
    scaler_y, per_currency_scalers, model = get_model_artifacts(model_name)
    X, feature_cols, df_proc = preprocess(_df_raw, _feat_cfg, per_currency_scalers, model=model)
    
    preds_scaled = model.predict(X).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_scaled).flatten()
    
    out_df = df_proc[["date", "currency_code", "exchange_rate"]].copy()
    out_df["predicted_rate"] = preds
    
    # Adjust dates based on target configuration
    target_mode = _feat_cfg.get("target", {}).get("mode", "raw_rate")
    n_ahead = _feat_cfg.get("target", {}).get("n_steps_ahead", 1)
    
    out_df["date"] = pd.to_datetime(out_df["date"])
    
    if target_mode == "n_step_ahead":
        # Shift the prediction dates into the future
        # (Assuming trading days, but we use calendar days for simplicity here)
        out_df["prediction_date"] = out_df["date"] + pd.Timedelta(days=n_ahead)
    else:
        out_df["prediction_date"] = out_df["date"]
        
    return out_df

# ── Sidebar UI ───────────────────────────────────────────────────────────────
st.sidebar.header("Dashboard Controls")

feat_cfg = get_config()
df_raw = get_raw_data()
currencies = sorted(df_raw["currency_code"].unique())

view_mode = st.sidebar.radio("View Mode", ["Single Currency", "Compare Currencies", "All"])

selected_currencies = []
if view_mode == "Single Currency":
    sel = st.sidebar.selectbox("Select Currency", currencies)
    selected_currencies = [sel] if sel else []
elif view_mode == "Compare Currencies":
    selected_currencies = st.sidebar.multiselect("Select Currencies", currencies, default=currencies[:2])
else:
    selected_currencies = currencies

model_opts = ["lgb", "xgb"]
selected_model = st.sidebar.selectbox("Model", model_opts, index=0)

recent_days = st.sidebar.slider("Present (Recent) Window Days", 10, 100, 30)

if not selected_currencies:
    st.warning("Please select at least one currency.")
    st.stop()

# ── Main Logic ───────────────────────────────────────────────────────────────
with st.spinner("Generating predictions..."):
    df_preds = get_predictions(selected_model, df_raw, feat_cfg)

# Filter by selected currencies
df_filtered = df_preds[df_preds["currency_code"].isin(selected_currencies)].copy()

if df_filtered.empty:
    st.warning("No data available for the selected configuration.")
    st.stop()

st.subheader(f"Model: `{selected_model.upper()}`")

def plot_currency(df_curr, code):
    df_curr = df_curr.sort_values("date").reset_index(drop=True)
    
    # Past vs Present boundaries
    max_date = df_curr["date"].max()
    present_cutoff = max_date - pd.Timedelta(days=recent_days)
    
    df_past = df_curr[df_curr["date"] < present_cutoff]
    df_present = df_curr[df_curr["date"] >= present_cutoff]
    
    # Future forecasts (predictions mapped to future dates beyond max_date)
    df_future = df_curr[df_curr["prediction_date"] > max_date]
    
    # Tabs for different views
    t1, t2, t3 = st.tabs(["Present (Recent) & Future", "Past (Historical)", "Data Table"])
    
    with t1:
        fig = go.Figure()
        # Actual Present
        fig.add_trace(go.Scatter(x=df_present["date"], y=df_present["exchange_rate"],
                                 mode='lines+markers', name='Actual', line=dict(color='blue')))
        # Predicted Present
        fig.add_trace(go.Scatter(x=df_present["prediction_date"], y=df_present["predicted_rate"],
                                 mode='lines', name='Predicted', line=dict(color='orange', dash='dash')))
        
        # Future predictions
        if not df_future.empty:
             fig.add_trace(go.Scatter(x=df_future["prediction_date"], y=df_future["predicted_rate"],
                                      mode='markers', name='Future Forecast', 
                                      marker=dict(color='red', size=8, symbol='star')))
             
        fig.update_layout(title=f"{code}: Recent Actual vs Predicted (Last {recent_days} days)",
                          xaxis_title="Date", yaxis_title="Exchange Rate", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
    with t2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_past["date"], y=df_past["exchange_rate"],
                                 mode='lines', name='Actual', line=dict(color='blue', width=1)))
        fig2.add_trace(go.Scatter(x=df_past["prediction_date"], y=df_past["predicted_rate"],
                                 mode='lines', name='Predicted', line=dict(color='orange', width=1, dash='dot')))
        fig2.update_layout(title=f"{code}: Historical Performance",
                           xaxis_title="Date", yaxis_title="Exchange Rate", height=500)
        st.plotly_chart(fig2, use_container_width=True)
        
    with t3:
        st.dataframe(df_curr.tail(100), use_container_width=True)


if view_mode in ["Single Currency", "All"]:
    for code in selected_currencies:
        st.markdown("---")
        st.markdown(f"### {code}")
        df_curr = df_filtered[df_filtered["currency_code"] == code]
        plot_currency(df_curr, code)
        
elif view_mode == "Compare Currencies":
    st.markdown("---")
    st.markdown("### Currencies Comparison (Present & Future)")
    fig_comp = go.Figure()
    
    max_d = df_filtered["date"].max()
    cut = max_d - pd.Timedelta(days=recent_days)
    df_recent = df_filtered[df_filtered["date"] >= cut]
    
    for code in selected_currencies:
        df_c = df_recent[df_recent["currency_code"] == code]
        # Normalize actuals and predictions to percentage change from the start of the window
        # for a fair comparison scale.
        if not df_c.empty:
            base_val = df_c["exchange_rate"].iloc[0]
            if base_val != 0:
                y_act = (df_c["exchange_rate"] - base_val) / base_val * 100
                y_pred = (df_c["predicted_rate"] - base_val) / base_val * 100
                
                fig_comp.add_trace(go.Scatter(x=df_c["date"], y=y_act, mode='lines', 
                                              name=f'{code} Actual (%)'))
                fig_comp.add_trace(go.Scatter(x=df_c["prediction_date"], y=y_pred, mode='lines', 
                                              name=f'{code} Predicted (%)', line=dict(dash='dot')))
                                              
    fig_comp.update_layout(title=f"Relative Performance (% Change over last {recent_days} days)",
                           xaxis_title="Date", yaxis_title="% Change", height=600)
    st.plotly_chart(fig_comp, use_container_width=True)
