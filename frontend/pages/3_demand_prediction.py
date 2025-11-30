import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Demand Prediction",
    page_icon="📈",
    layout="wide"
)

# Dark mode CSS
st.markdown("""
    <style>
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0E1117 !important;
        color: #E6EDF3 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #00D9FF !important;
    }
    .stMetric {
        background-color: #161B22 !important;
        border: 1px solid rgba(0, 217, 255, 0.1);
        border-radius: 8px;
        padding: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <div style='font-size: 3em; margin-bottom: 10px;'>📈</div>
        <h1 style='color: #00D9FF;'>Demand Prediction</h1>
        <p style='color: #E6EDF3; font-size: 1.1em;'>SARIMA-Based Employee Growth Forecast</p>
    </div>
""", unsafe_allow_html=True)

st.divider()

# ============ LOAD DATA ============

@st.cache_resource
def load_data():
    """Load company growth data"""
    
    df = pd.read_csv("../data/company_growth_detailed.csv")
    
    df['date'] = pd.to_datetime(df['date'])
    return df

# ============ LOAD SARIMA MODEL ============

@st.cache_resource
def load_sarima_model():
    """Load pre-trained SARIMA model"""

    model_dir ='../backend/models'
    
    if model_dir is None:
        st.error("""
            ❌ SARIMA model not found!
            
            Please train the model first:
            ```
            cd backend
            python train_timeseries_model.py
            ```
        """)
        st.stop()
    
    try:
        with open(os.path.join(model_dir, 'sarima_model.pkl'), 'rb') as f:
            model = pickle.load(f)
        
        with open(os.path.join(model_dir, 'sarima_metadata.json'), 'r') as f:
            metadata = json.load(f)
        
        return model, metadata
    except Exception as e:
        st.error(f"Error loading SARIMA model: {str(e)}")
        st.stop()

# Load data and model
df = load_data()
sarima_model, metadata = load_sarima_model()

#st.success(f"✅ Loaded data: {len(df)} months ({df['date'].iloc[0].strftime('%Y-%m')} to {df['date'].iloc[-1].strftime('%Y-%m')})")
st.info(f"📊 Model: {metadata['model_name']} | MAE: ±{metadata['test_mae']:.0f} employees")

# ============ GENERATE FORECAST ============

@st.cache_data
def generate_forecast(_sarima_model, _df):
    """Generate 6-month forecast using pre-trained model"""
    
    # Get forecast
    forecast_result = _sarima_model.get_forecast(steps=6)
    forecast_values = np.array(forecast_result.predicted_mean)
    forecast_ci = forecast_result.conf_int()
    
    # Extract confidence intervals
    if isinstance(forecast_ci, np.ndarray):
        lower_ci = forecast_ci[:, 0]
        upper_ci = forecast_ci[:, 1]
    else:
        lower_ci = forecast_ci.iloc[:, 0].values
        upper_ci = forecast_ci.iloc[:, 1].values
    
    # Ensure monotonic increase
    last_value = _df['employees'].iloc[-1]
    forecast_values = np.maximum(forecast_values, last_value)
    for i in range(1, len(forecast_values)):
        if forecast_values[i] < forecast_values[i-1]:
            forecast_values[i] = forecast_values[i-1]
    
    forecast_values = np.round(forecast_values).astype(int)
    lower_ci = np.round(lower_ci).astype(int)
    upper_ci = np.round(upper_ci).astype(int)
    
    # Create forecast dates
    last_date = _df['date'].iloc[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=6, freq='MS')
    
    df_forecast = pd.DataFrame({
        'date': forecast_dates,
        'month': forecast_dates.strftime('%Y-%m'),
        'forecast': forecast_values,
        'lower_ci': lower_ci,
        'upper_ci': upper_ci
    })
    
    return df_forecast, _df['employees'].iloc[-1]

# Generate forecast
df_forecast, last_value = generate_forecast(sarima_model, df)

# ============ KEY METRICS ============

st.markdown("## 📊 Key Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📍 Current",
        f"{last_value:,}",
        help="Last recorded value (Dec 2024)"
    )

with col2:
    predicted = df_forecast['forecast'].iloc[-1]
    growth = predicted - last_value
    st.metric(
        "🔮 6-Month Forecast",
        f"{predicted:,}",
        delta=f"+{growth:,}" if growth >= 0 else f"{growth:,}"
    )

with col3:
    growth_rate = ((predicted / last_value) - 1) * 100 if last_value > 0 else 0
    st.metric(
        "📈 Expected Growth",
        f"{growth_rate:.2f}%"
    )

with col4:
    avg_monthly = growth / 6
    st.metric(
        "📅 Avg Monthly",
        f"+{avg_monthly:.0f}"
    )

st.divider()

# ============ MAIN FORECAST CHART ============

st.markdown("## 📈 Historical Data + 6-Month SARIMA Forecast")

fig = go.Figure()

# Historical data
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['employees'],
    name='Historical Data',
    mode='lines+markers',
    line=dict(color='#00D9FF', width=3),
    marker=dict(size=4),
    hovertemplate='<b>%{x|%Y-%m}</b><br>Employees: %{y:,.0f}<extra></extra>'
))

# Forecast
fig.add_trace(go.Scatter(
    x=df_forecast['date'],
    y=df_forecast['forecast'],
    name='6-Month Forecast',
    mode='lines+markers',
    line=dict(color='#FFC000', width=3, dash='dash'),
    marker=dict(size=6, symbol='diamond'),
    hovertemplate='<b>%{x|%Y-%m}</b><br>Forecast: %{y:,.0f}<extra></extra>'
))

# Confidence interval
fig.add_trace(go.Scatter(
    x=list(df_forecast['date']) + list(df_forecast['date'][::-1]),
    y=list(df_forecast['upper_ci']) + list(df_forecast['lower_ci'][::-1]),
    fill='toself',
    name='95% Confidence Interval',
    line=dict(color='rgba(255, 192, 0, 0)'),
    fillcolor='rgba(255, 192, 0, 0.2)',
    hoverinfo='skip'
))

fig.update_layout(
    height=500,
    hovermode='x unified',
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(22,27,34,0.5)",
    font=dict(color="#E6EDF3", size=12),
    xaxis=dict(
        showgrid=True, 
        gridwidth=1, 
        gridcolor="rgba(88,166,255,0.1)",
        title="Date"
    ),
    yaxis=dict(
        showgrid=True, 
        gridwidth=1, 
        gridcolor="rgba(88,166,255,0.1)",
        title="Number of Employees"
    ),
    legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.7)"),
    #title="Company Employee Growth: Historical + 6-Month SARIMA Forecast"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============ FORECAST TABLE ============

st.markdown("## 📅 6-Month Forecast Details")

forecast_display = df_forecast[['month', 'forecast', 'lower_ci', 'upper_ci']].copy()
forecast_display.columns = ['Month', 'Forecast', 'Lower 95% CI', 'Upper 95% CI']
forecast_display['Monthly Growth'] = forecast_display['Forecast'].diff().fillna(0).astype(int)

st.dataframe(
    forecast_display,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Forecast": st.column_config.NumberColumn(format="%d"),
        "Lower 95% CI": st.column_config.NumberColumn(format="%d"),
        "Upper 95% CI": st.column_config.NumberColumn(format="%d"),
        "Monthly Growth": st.column_config.NumberColumn(format="%+d")
    }
)

st.divider()

# ============ STATISTICS ============

st.markdown("## 📊 Forecast Statistics")

col_stat1, col_stat2, col_stat3 = st.columns(3)

with col_stat1:
    st.markdown("### Historical Data")
    st.write(f"**Records:** {len(df)} months")
    st.write(f"**Start:** {df['date'].iloc[0].strftime('%Y-%m')}")
    st.write(f"**End:** {df['date'].iloc[-1].strftime('%Y-%m')}")
    st.write(f"**Total Growth:** {df['employees'].iloc[-1] - df['employees'].iloc[0]:,}")

with col_stat2:
    st.markdown("### Forecast Period")
    st.write(f"**Horizon:** 6 months")
    st.write(f"**Start:** {df_forecast['month'].iloc[0]}")
    st.write(f"**End:** {df_forecast['month'].iloc[-1]}")
    st.write(f"**Forecast Growth:** {df_forecast['forecast'].iloc[-1] - last_value:,}")

with col_stat3:
    st.markdown("### Model")
    st.write(f"**Type:** {metadata['model_name']}")
    st.write(f"**Order:** {metadata['order']}")
    st.write(f"**Seasonal:** {metadata['seasonal_order']}")
    st.write(f"**Confidence:** 95%")

st.divider()

# ============ DOWNLOAD ============

st.markdown("## 💾 Export Forecast")

# Combine historical and forecast
df_combined = pd.DataFrame({
    'date': pd.concat([df['date'], df_forecast['date']], ignore_index=True),
    'month': pd.concat([df['month'], df_forecast['month']], ignore_index=True),
    'type': ['actual'] * len(df) + ['forecast'] * len(df_forecast),
    'employees': pd.concat([df['employees'], df_forecast['forecast']], ignore_index=True)
})

csv = df_combined.to_csv(index=False)

st.download_button(
    label="📥 Download Forecast Data (CSV)",
    data=csv,
    file_name=f"sarima_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv"
)