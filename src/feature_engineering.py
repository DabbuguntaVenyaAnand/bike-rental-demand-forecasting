"""Dataset loading, temporal splitting and leakage-safe feature engineering."""

from pathlib import Path

import numpy as np
import pandas as pd

from model_config import (
    ALL_FEATURES,
    DATA_PATH,
    LEAKAGE_COLUMNS,
    SEASON_MAP,
    WEATHER_MAP,
    WEEKDAY_PEAK_HOURS,
    WEEKEND_PEAK_HOURS,
)


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Place UCI hour.csv in data/hour.csv."
        )

    df = pd.read_csv(path, parse_dates=["dteday"])
    return df.sort_values(["dteday", "hr"]).dropna().reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar and cyclic features without target-derived information."""
    out = df.copy()

    out["season_label"] = out["season"].map(SEASON_MAP)
    out["weathersit_label"] = out["weathersit"].map(WEATHER_MAP)

    out["hour_sin"] = np.sin(2 * np.pi * out["hr"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hr"] / 24.0)

    # UCI Bike Sharing coding: Sunday=0, ..., Saturday=6.
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["weekday"] / 7.0)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["weekday"] / 7.0)

    out["month_sin"] = np.sin(2 * np.pi * (out["mnth"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["mnth"] - 1) / 12.0)

    out["is_weekend"] = out["weekday"].isin([0, 6]).astype(int)

    out["is_rush_hour"] = (
        (out["workingday"] == 1) & out["hr"].isin(WEEKDAY_PEAK_HOURS)
    ).astype(int)

    weekday_peak = (
        (out["workingday"] == 1) & out["hr"].isin(WEEKDAY_PEAK_HOURS)
    )
    weekend_peak = (
        (out["workingday"] == 0) & out["hr"].isin(WEEKEND_PEAK_HOURS)
    )

    out["peak_time"] = np.select(
        [weekday_peak, out["workingday"].eq(1), weekend_peak],
        ["weekday_peak", "weekday_off", "weekend_peak"],
        default="weekend_off",
    )

    return out


def temporal_split(df: pd.DataFrame, test_size: float = 0.20):
    """Use the latest ~20% of dates as the untouched holdout set."""
    split_day = df["dteday"].quantile(1 - test_size)

    train = df.loc[df["dteday"] <= split_day].copy()
    test = df.loc[df["dteday"] > split_day].copy()

    return train, test, split_day


def feature_matrix(df: pd.DataFrame):
    engineered = add_engineered_features(df)
    X = engineered[ALL_FEATURES].copy()
    y = engineered["cnt"].astype(float).copy()
    return X, y


def validate_design(df: pd.DataFrame, X: pd.DataFrame) -> None:
    """Fail early if future edits introduce leakage or ordering problems."""
    leakage = LEAKAGE_COLUMNS & set(X.columns)
    if leakage:
        raise ValueError(
            f"Target-leakage columns present in features: {sorted(leakage)}"
        )

    if X.isna().any().any():
        raise ValueError("Feature matrix contains missing values.")

    ordered = df.sort_values(["dteday", "hr"]).reset_index(drop=True)
    current = df.reset_index(drop=True)

    if not current[["dteday", "hr"]].equals(ordered[["dteday", "hr"]]):
        raise ValueError("Dataset must be chronologically sorted before splitting.")
