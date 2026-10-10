"""Streamlit demo for Bike Rental Demand Forecasting.

Run from the repository root:
    streamlit run app/demo_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

# Allow imports from src/ when Streamlit launches this file from app/.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feature_engineering import add_engineered_features
from model_config import ALL_FEATURES


MODEL_PATH = ROOT / "models" / "production_model.pkl"
DIAGNOSTICS_PATH = ROOT / "models" / "model_diagnostics.json"

SEASON_TO_CODE = {
    "Winter": 1,
    "Spring": 2,
    "Summer": 3,
    "Fall": 4,
}

WEATHER_TO_CODE = {
    "Clear / Few clouds": 1,
    "Mist / Cloudy": 2,
    "Light rain / Light snow": 3,
    "Heavy rain / Snow / Fog": 4,
}

# UCI Bike Sharing dataset coding:
# Sunday = 0, Monday = 1, ..., Saturday = 6.
WEEKDAY_TO_CODE = {
    "Sunday": 0,
    "Monday": 1,
    "Tuesday": 2,
    "Wednesday": 3,
    "Thursday": 4,
    "Friday": 5,
    "Saturday": 6,
}

YEAR_TO_CODE = {
    2011: 0,
    2012: 1,
}


def normalize_atemp(celsius: float) -> float:
    """Normalize feels-like temperature exactly as documented by UCI."""
    # atemp = (t - (-16)) / (50 - (-16))
    return float(np.clip((celsius + 16.0) / 66.0, 0.0, 1.0))


def normalize_humidity(percent: int) -> float:
    return float(np.clip(percent / 100.0, 0.0, 1.0))


def normalize_windspeed(kmh: float) -> float:
    return float(np.clip(kmh / 67.0, 0.0, 1.0))


def derive_workingday(weekday: int, holiday: bool) -> int:
    """UCI workingday = 1 only when the day is neither weekend nor holiday."""
    is_weekend = weekday in (0, 6)
    return int((not is_weekend) and (not holiday))


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def selected_model_name() -> str:
    if not DIAGNOSTICS_PATH.exists():
        return "trained production model"

    try:
        payload = json.loads(DIAGNOSTICS_PATH.read_text(encoding="utf-8"))
        return payload.get("selected_model", "trained production model")
    except (OSError, json.JSONDecodeError):
        return "trained production model"


st.set_page_config(
    page_title="Bike Rental Demand Forecasting",
    page_icon="🚲",
    layout="wide",
)

st.title("🚲 Bike Rental Demand Forecasting")
st.caption(
    "Enter a riding scenario to estimate the expected number of bike rentals "
    "for that hour."
)

with st.sidebar:
    st.header("Calendar & weather")

    year = st.selectbox("Year", (2011, 2012), index=1)
    month = st.slider("Month", 1, 12, 7)

    weekday_label = st.selectbox(
        "Day of week",
        tuple(WEEKDAY_TO_CODE.keys()),
        index=5,  # Friday
    )

    holiday = st.checkbox("Public holiday", value=False)

    season_label = st.radio(
        "Season",
        tuple(SEASON_TO_CODE.keys()),
        index=2,
    )

    weather_label = st.selectbox(
        "Weather condition",
        tuple(WEATHER_TO_CODE.keys()),
        index=0,
    )

weekday = WEEKDAY_TO_CODE[weekday_label]
workingday = derive_workingday(weekday, holiday)

col1, col2, col3 = st.columns(3)

with col1:
    feels_like = st.slider(
        "Feels-like temperature (°C)",
        -16.0,
        50.0,
        27.0,
        0.5,
    )
    hour = st.slider("Hour of day", 0, 23, 17)

with col2:
    humidity = st.slider("Humidity (%)", 0, 100, 55)
    st.metric("Derived working-day status", "Yes" if workingday else "No")

with col3:
    windspeed = st.slider("Windspeed (km/h)", 0.0, 67.0, 12.0, 0.5)
    st.caption(
        "Working-day status is derived automatically from weekday + holiday."
    )

st.markdown("---")

# Build a row in the original UCI feature representation first.
raw_input = pd.DataFrame(
    {
        "season": [SEASON_TO_CODE[season_label]],
        "yr": [YEAR_TO_CODE[year]],
        "mnth": [month],
        "hr": [hour],
        "holiday": [int(holiday)],
        "weekday": [weekday],
        "workingday": [workingday],
        "weathersit": [WEATHER_TO_CODE[weather_label]],
        "atemp": [normalize_atemp(feels_like)],
        "hum": [normalize_humidity(humidity)],
        "windspeed": [normalize_windspeed(windspeed)],
    }
)

# Reuse the exact feature engineering used during model training.
engineered = add_engineered_features(raw_input)
X_input = engineered[ALL_FEATURES].copy()

if st.button("Predict demand", type="primary"):
    st.session_state["prediction_input"] = X_input

st.subheader("Model response")

pending = st.session_state.get("prediction_input")
model = load_model()

if pending is None:
    st.info("Adjust the inputs and click **Predict demand**.")
elif model is None:
    st.warning(
        "No final trained model was found at `models/production_model.pkl`. "
        "Run `python src/train_models.py` from the repository root first."
    )
else:
    prediction = float(np.clip(model.predict(pending)[0], 0, None))

    st.metric(
        "Predicted hourly bike rentals",
        f"{prediction:.0f}",
    )

    st.caption(f"Prediction generated using: **{selected_model_name()}**")

    with st.expander("Show engineered feature vector sent to the model"):
        st.dataframe(pending.T, use_container_width=True)

st.markdown("---")
st.subheader("What drives the prediction?")
st.write(
    "The model combines time-of-day and calendar patterns with weather-related "
    "conditions such as feels-like temperature, humidity and windspeed. "
    "The final production model was selected using temporal cross-validation "
    "with RMSLE as the primary evaluation metric."
)
