"""Streamlit demo for Bike Rental Demand Forecasting.

Run from the repository root:
    streamlit run app/demo_app.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
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

WEATHER_TO_CODE = {
    "Clear / Few clouds": 1,
    "Mist / Cloudy": 2,
    "Light rain / Light snow": 3,
    "Heavy rain / Snow / Fog": 4,
}

SEASON_LABELS = {
    1: "Winter",
    2: "Spring",
    3: "Summer",
    4: "Fall",
}

YEAR_TO_CODE = {
    2011: 0,
    2012: 1,
}

WEEKDAY_LABELS = {
    0: "Sunday",
    1: "Monday",
    2: "Tuesday",
    3: "Wednesday",
    4: "Thursday",
    5: "Friday",
    6: "Saturday",
}


def uci_weekday(selected_date: date) -> int:
    """Convert Python Monday=0..Sunday=6 to UCI Sunday=0..Saturday=6."""
    return (selected_date.weekday() + 1) % 7


def derive_season(selected_date: date) -> int:
    """
    Derive the season code used by the Bike Sharing dataset.

    Boundaries mirror the seasonal transitions present in the 2011–2012 data:
    Winter: Dec 21 – Mar 20
    Spring: Mar 21 – Jun 20
    Summer: Jun 21 – Sep 22
    Fall:   Sep 23 – Dec 20
    """
    md = (selected_date.month, selected_date.day)

    if (3, 21) <= md < (6, 21):
        return 2
    if (6, 21) <= md < (9, 23):
        return 3
    if (9, 23) <= md < (12, 21):
        return 4
    return 1


def normalize_atemp(celsius: float) -> float:
    """Normalize feels-like temperature using the UCI atemp definition."""
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
    "Estimate hourly bike rental demand from calendar and weather conditions."
)

with st.sidebar:
    st.header("Scenario")

    selected_date = st.date_input(
        "Date",
        value=date(2012, 7, 6),
        min_value=date(2011, 1, 1),
        max_value=date(2012, 12, 31),
    )

    holiday = st.checkbox(
        "Public holiday",
        value=False,
        help="Check this only if the selected date is a public holiday.",
    )

    weather_label = st.selectbox(
        "Weather condition",
        tuple(WEATHER_TO_CODE.keys()),
        index=0,
    )


year = selected_date.year
month = selected_date.month
weekday = uci_weekday(selected_date)
season = derive_season(selected_date)
workingday = derive_workingday(weekday, holiday)

st.subheader("Derived calendar features")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Day", WEEKDAY_LABELS[weekday])
c2.metric("Month", selected_date.strftime("%B"))
c3.metric("Season", SEASON_LABELS[season])
c4.metric("Working day", "Yes" if workingday else "No")

st.caption(
    "Calendar features are derived automatically from the selected date to keep "
    "the model inputs internally consistent."
)

st.markdown("---")

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

with col3:
    windspeed = st.slider("Windspeed (km/h)", 0.0, 67.0, 12.0, 0.5)

st.markdown("---")

# Build a row in the original UCI feature representation first.
raw_input = pd.DataFrame(
    {
        "season": [season],
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
    st.info("Adjust the scenario and click **Predict demand**.")
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
st.subheader("About the prediction")
st.write(
    "The model combines temporal patterns with weather-related conditions such "
    "as feels-like temperature, humidity and windspeed. The production model "
    "was selected using temporal cross-validation with RMSLE as the primary "
    "evaluation metric."
)
