"""Shared configuration for model training and evaluation."""

from pathlib import Path

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hour.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

SEASON_MAP = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
WEATHER_MAP = {1: "Clear", 2: "Cloudy", 3: "Wet", 4: "Severe"}

WEEKDAY_PEAK_HOURS = (7, 8, 9, 17, 18, 19)
WEEKEND_PEAK_HOURS = tuple(range(10, 19))

NUMERIC_FEATURES = [
    "atemp",
    "hum",
    "windspeed",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
]

CATEGORICAL_FEATURES = [
    "season_label",
    "weathersit_label",
    "hr",
    "weekday",
    "mnth",
    "yr",
    "holiday",
    "workingday",
    "peak_time",
    "is_rush_hour",
    "is_weekend",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LEAKAGE_COLUMNS = {"cnt", "casual", "registered", "instant"}
