import os
import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

os.environ["HTTP_PROXY"] = ""
os.environ["HTTPS_PROXY"] = ""
os.environ["NO_PROXY"] = "*"

st.set_page_config(
    page_title="Abuja Rain Predictor",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-title { font-size: 1.1rem; color: #4B5563; margin-bottom: 2rem; }
    .metric-card { background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6; }
    .prediction-box { padding: 20px; border-radius: 10px; text-align: center; font-size: 1.5rem; font-weight: bold; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

ABUJA_LAT = 9.05
ABUJA_LON = 7.32


@st.cache_data(show_spinner="📥 Connecting to Open-Meteo Archive & loading Abuja weather data...")
def fetch_historical_data():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": ABUJA_LAT,
        "longitude": ABUJA_LON,
        "start_date": "2023-01-01",
        "end_date": "2025-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "rain_sum",
            "wind_speed_10m_max",
            "et0_fao_evapotranspiration"
        ],
        "timezone": "Africa/Lagos"
    }

    try:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            st.error(f"⚠️ API Connection Error (Status {response.status_code})")
            return None

        data = response.json()
        daily_data = data.get("daily", {})

        if not daily_data:
            return None

        df = pd.DataFrame(daily_data)
        if 'time' in df.columns:
            df = df.rename(columns={'time': 'date'})

        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Connection failed: {e}")
        return None


@st.cache_resource(show_spinner="Optimizing Artificial Intelligence Model...")
def train_pipeline(df):
    df['will_rain'] = (df['rain_sum'] > 0.1).astype(int)

    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear

    feature_cols = [
        'temperature_2m_max',
        'temperature_2m_min',
        'wind_speed_10m_max',
        'et0_fao_evapotranspiration',
        'month',
        'day_of_year'
    ]

    X = df[feature_cols].fillna(df[feature_cols].mean())
    y = df['will_rain']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    monthly_averages = df.groupby('month')[feature_cols[:-2]].mean().to_dict(orient='index')

    return model, acc, feature_cols, monthly_averages


st.markdown("<div class='main-title'>🌧️ Abuja Smart Rain Predictor</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>An interactive Machine Learning system forecasting precipitation intervals for Abuja, Nigeria. </div>", unsafe_allow_html=True)
st.markdown("   ")
st.subheader("About")
st.markdown("<div class='about-box'> This intelligent web application serves as a Machine Learning Capstone Project designed to forecast precipitation events specifically within Abuja, Nigeria. By establishing a real-time data pipeline with the Open-Meteo Archive API, the system dynamically harvests three full years of historical regional weather parameters. An advanced Random Forest Classifier trains on the fly to accurately map complex atmospheric behaviors against Abuja's distinct wet and dry seasonal cycles. Users can effortlessly input any target calendar date to receive instant, data-driven binary classifications accompanied by mathematical probability metrics. Ultimately, this project highlights how modern cloud-deployed AI frameworks can successfully democratize access to localized predictive meteorology.</div>", unsafe_allow_html=True)
st.markdown("   ")

df_raw = fetch_historical_data()

if df_raw is not None:
    model, accuracy, feature_cols, seasonal_defaults = train_pipeline(df_raw)


    st.subheader("Select Your Target Date")
    target_date = st.date_input("Choose any date to predict weather parameters for:", datetime.today())

    selected_month = target_date.month
    selected_doy = target_date.timetuple().tm_yday

    # Extract automatic climate baseline values based on what is historical for that month
    defaults = seasonal_defaults.get(selected_month, {'temperature_2m_max': 31.0, 'temperature_2m_min': 22.0,
                                                      'wind_speed_10m_max': 12.0, 'et0_fao_evapotranspiration': 4.5})

    st.write("")

    with st.expander("Advanced Meteorological Overrides (Optional Calibration)"):
        st.write(
            "The system has automatically filled these fields using historical monthly defaults for Abuja. Adjust them if you want to test micro-climate shifts:")
        col_a, col_b = st.columns(2)
        with col_a:
            t_max = st.slider("Max Estimated Temperature (°C)", 15.0, 45.0, float(defaults['temperature_2m_max']),
                              step=0.5)
            t_min = st.slider("Min Estimated Temperature (°C)", 10.0, 35.0, float(defaults['temperature_2m_min']),
                              step=0.5)
        with col_b:
            w_max = st.slider("Max Wind Speed (km/h)", 0.0, 50.0, float(defaults['wind_speed_10m_max']), step=0.5)
            evap = st.slider("Evapotranspiration rate (mm)", 0.0, 15.0, float(defaults['et0_fao_evapotranspiration']),
                             step=0.1)

    st.write("---")
    st.subheader("Generate Prediction Output")

    input_features = pd.DataFrame([{
        'temperature_2m_max': t_max if 't_max' in locals() else defaults['temperature_2m_max'],
        'temperature_2m_min': t_min if 't_min' in locals() else defaults['temperature_2m_min'],
        'wind_speed_10m_max': w_max if 'w_max' in locals() else defaults['wind_speed_10m_max'],
        'et0_fao_evapotranspiration': evap if 'evap' in locals() else defaults['et0_fao_evapotranspiration'],
        'month': selected_month,
        'day_of_year': selected_doy
    }])[feature_cols]

    if st.button("Analyze Atmospheric Data", type="primary", use_container_width=True):
        prediction = model.predict(input_features)[0]
        probabilities = model.predict_proba(input_features)[0]

        if prediction == 1:
            st.markdown(f"""
                <div class='prediction-box' style='background-color: #FEE2E2; color: #991B1B;'>
                    🌧️ Prediction: Expect Rain on this Day!
                    <div style='font-size: 1rem; font-weight: normal; margin-top: 5px; color: #7F1D1D;'>
                        Confidence Metrics: There is a {probabilities[1]:.1%} dynamic mathematical probability of local rainfall events.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='prediction-box' style='background-color: #DCFCE7; color: #166534;'>
                    ☀️ Prediction: Safe, Dry Conditions Foreseeable!
                    <div style='font-size: 1rem; font-weight: normal; margin-top: 5px; color: #14532D;'>
                        Confidence Metrics: There is a {probabilities[0]:.1%} probability of zero/insignificant rainfall.
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.write("")
    with st.expander("📋 View Underlying Active Ground-Truth Dataset"):
        st.dataframe(
            df_raw[['date', 'temperature_2m_max', 'temperature_2m_min', 'rain_sum', 'wind_speed_10m_max']].tail(10),
            use_container_width=True)

else:
    st.error(
        "System pipeline initialized but failed to securely sync data components. Please review your active internet connection.")
