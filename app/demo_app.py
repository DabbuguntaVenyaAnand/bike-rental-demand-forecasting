"""Bike Rental Demand Forecasting - demo input/interface side.

Streamlit app (Person A) where a user enters rental/riding conditions and
requests a demand prediction. If Person B's production model exists it is used,
otherwise the Person A baseline pipeline handles the inference.

Run:
    .venv/Scripts/streamlit run app/demo_app.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Bike Demand — Inputs", page_icon="🚲", layout="wide")

st.title("🚲 Bike Rental Demand — Prediction Inputs")
st.caption(
    "Person A: input/interface side. Enter the riding scenario, review the "
    "preprocessing summary, and send it to the trained model for a prediction."
)

sidebar = st.sidebar

season_label = sidebar.radio(
    "Season",
    ("Winter", "Spring", "Summer", "Fall"),
    index=2,
)
year = sidebar.selectbox("Year", (2011, 2012), index=1)
month = sidebar.slider("Month", 1, 12, 7)
day_of_week = sidebar.selectbox(
    "Day of week",
    ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
    index=4,
)
holiday = sidebar.checkbox("Public holiday", value=False)
workingday = sidebar.checkbox("Working day (Mon-Fri, non-holiday)", value=True)
weather_label = sidebar.selectbox(
    "Weather condition",
    ("Clear", "Cloudy", "Wet", "Severe"),
    index=0,
)

cols = st.columns(3)
atemp = cols[0].slider("Feels-like temperature (°C)", -10.0, 50.0, 27.0, 0.5)
humidity = cols[1].slider("Humidity (%)", 0, 100, 55)
windspeed = cols[2].slider("Windspeed (km/h)", 0.0, 67.0, 12.0, 0.5)
hr = cols[0].slider("Hour of day (0-23)", 0, 23, 17)
temp = cols[1].slider("Air temperature (°C)", -5.0, 41.0, 24.0, 0.5)  # shown for realism; NOT sent to model (paper: drop raw temp)

st.markdown("---")

season_map = {"Winter": 1, "Spring": 2, "Summer": 3, "Fall": 4}
dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
weather_map = {"Clear": 1, "Cloudy": 2, "Wet": 3, "Severe": 4}
yr_map = {2011: 0, 2012: 1}

WEEKDAY_PEAK_HOURS = (7, 8, 9, 17, 18, 19)
WEEKEND_PEAK_HOURS = tuple(range(10, 19))
temp_bucket = None


def _peak_bucket(working: int, hour: int) -> str:
    if working == 1:
        if hour in WEEKDAY_PEAK_HOURS:
            return "weekday_peak"
        return "weekday_off"
    if hour in WEEKEND_PEAK_HOURS:
        return "weekend_peak"
    return "weekend_off"


# Paper-aligned featurization (temp dropped, peak buckets, temp buckets)
# Compute the same atemp quantile bin edges the training script's qcut used,
# so app-side bucketing matches training exactly.
TEMP_BUCKET_LABELS = ["cold", "mild", "warm", "hot"]

@st.cache_data
def _temp_bucket_edges() -> list[float]:
    data_path = ROOT / "data" / "hour.csv"
    atemp_series = pd.read_csv(data_path, usecols=["atemp"])["atemp"]
    _, edges = pd.qcut(atemp_series, q=4, labels=TEMP_BUCKET_LABELS, retbins=True, duplicates="drop")
    return [float(e) for e in edges]


def _temp_bucket(atemp_norm: float) -> str:
    edges = _temp_bucket_edges()
    atemp_raw = atemp_norm * 50.0
    idx = 0
    for i, edge in enumerate(edges[1:-1]):  # interior edges (3 for q=4)
        if atemp_raw >= edge:
            idx = i + 1
    return TEMP_BUCKET_LABELS[idx]


temp_bucket = _temp_bucket(atemp / 50.0)

X_input = pd.DataFrame(
    {
        "atemp": [atemp / 50.0],
        "hum": [humidity / 100.0],
        "windspeed": [windspeed / 67.0],
        "hour_sin": [float(np.sin(2 * np.pi * hr / 24.0))],
        "hour_cos": [float(np.cos(2 * np.pi * hr / 24.0))],
        "dayofweek_sin": [float(np.sin(2 * np.pi * dow_map[day_of_week] / 7.0))],
        "dayofweek_cos": [float(np.cos(2 * np.pi * dow_map[day_of_week] / 7.0))],
        "month_sin": [float(np.sin(2 * np.pi * (month - 1) / 12.0))],
        "month_cos": [float(np.cos(2 * np.pi * (month - 1) / 12.0))],
        "is_rush_hour": [int(hr in WEEKDAY_PEAK_HOURS)],
        "is_weekend": [int(day_of_week in ("Sat", "Sun"))],
        "peak_time": [_peak_bucket(int(workingday), hr)],
        "temp_bucket": [temp_bucket],
        "season_label": [season_label],
        "weathersit_label": [weather_label],
        "mnth": [month],
        "yr": [yr_map[year]],
        "holiday": [int(holiday)],
        "workingday": [int(workingday)],
    }
)

if st.button("Predict demand", type="primary"):
    st.session_state["last_input"] = X_input

st.subheader("Model response")
X_pending = st.session_state.get("last_input")
if X_pending is None:
    st.info("Adjust the inputs and click **Predict demand** to send them to the model.")
else:
    model_path = ROOT / "models" / "baseline_model.pkl"
    if not model_path.exists():
        st.warning(
            "No trained model artifact found. Run "
            "`src/preprocess_and_baseline.py` first to create it."
        )
    else:
        import joblib

        pipe = joblib.load(model_path)
        pred = float(np.clip(pipe.predict(X_pending)[0], 0, None))
        st.metric("Predicted hourly bike rentals", f"{pred:.0f}")

        with st.expander("Show the preprocessed input vector sent to the model"):
            st.write(X_pending.T)

st.markdown("---")
st.subheader("Why these inputs matter")
st.caption(
    "Hour of day and working day drive the dual rush-hour peaks (see "
    "`reports/03_hourly_box.png` and the paper-style peak buckets in "
    "`reports/12_workingday_hourly_profile.png`). Feels-like temperature, "
    "humidity and weather situation push demand up or down "
    "(see `reports/09_correlation_heatmap.png`). Air temperature is shown "
    "for context but deliberately excluded from the model per the reference "
    "paper's collinearity analysis."
)
