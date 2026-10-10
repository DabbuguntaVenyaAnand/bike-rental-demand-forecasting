"""Bike Rental Demand Forecasting - Person A.

Data understanding, EDA figures, preprocessing + feature engineering,
baseline models, and a metrics report.

Run:
    python src/data_understanding.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import json

sns.set_theme(style="whitegrid", context="talk")
RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hour.csv"
FIG_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

FIG_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

SEASON_MAP = {1: "Winter", 2: "Spring", 3: "Summer", 4: "Fall"}
WEATHER_MAP = {
    1: "Clear",
    2: "Cloudy",
    3: "Wet",
    4: "Severe",
}


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["dteday"])
    print(f"Loaded {len(df):,} hourly records from {DATA_PATH}")
    return df


def data_understanding(df: pd.DataFrame) -> None:
    print("\n=== SCHEMA ===")
    print(df.dtypes)
    print("\n=== MISSING VALUES (total) ===")
    print(df.isna().sum())
    print("\n=== DUPLICATED ROWS ===", df.duplicated().sum())
    print("\n=== DATE RANGE ===")
    print(df["dteday"].min(), "->", df["dteday"].max())
    print("\n=== TARGET STATISTICS ===")
    print(df["cnt"].describe().round(2))
    print("Skew:", round(df["cnt"].skew(), 3))


def pearson(x: pd.Series, y: pd.Series) -> float:
    return float(x.corr(y))


def relationship_eda(df: pd.DataFrame) -> None:
    for a, b, note in [
        ("temp", "cnt", "raw temp (reference only - highly collinear with atemp)"),
        ("atemp", "cnt", "feels-like temp (kept for modeling)"),
        ("hum", "cnt", "humidity (negative driver)"),
        ("windspeed", "cnt", "windspeed (weak driver)"),
    ]:
        print(f"correlation({a}, cnt) = {pearson(df[a], df[b]):.3f}   # {note}")


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE from the Kaggle competition the reference paper used."""
    return float(
        np.sqrt(
            np.mean(
                (np.log1p(y_pred) - np.log1p(y_true)) ** 2
            )
        )
    )


def score_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmsle": compute_rmsle(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def make_figures(df: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []

    def save(fig: plt.Figure, name: str) -> Path:
        p = FIG_DIR / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
        print("saved", p.name)
        return p

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["cnt"], bins=50, ax=ax)
    ax.set_title("Distribution of hourly rentals (cnt)")
    save(fig, "01_target_distribution.png")

    fig, ax = plt.subplots(figsize=(14, 5))
    daily = df.groupby("dteday")["cnt"].mean()
    ax.plot(daily.index, daily.values, color="tab:blue")
    ax.set_title("Average daily rentals over time")
    ax.set_ylabel("cnt")
    save(fig, "02_daily_trend.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x="hr", y="cnt", ax=ax)
    ax.set_title("Rentals by hour of day")
    save(fig, "03_hourly_box.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="season", y="cnt", ax=ax)
    ax.set_xticklabels(["Winter", "Spring", "Summer", "Fall"])
    ax.set_title("Average rentals per season")
    ax.set_xlabel("")
    save(fig, "04_season_avg.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=df, x="weathersit", y="cnt", ax=ax)
    ax.set_xticklabels(["Clear", "Cloudy", "Wet", "Severe"])
    ax.set_title("Average rentals by weather situation")
    save(fig, "05_weather_avg.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="atemp", y="cnt", alpha=0.25, ax=ax)
    ax.set_title("Rentals vs feels-like temperature (atemp)")
    save(fig, "06_atemp_scatter.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="hum", y="cnt", alpha=0.25, ax=ax)
    ax.set_title("Rentals vs humidity (hum)")
    save(fig, "07_hum_scatter.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="windspeed", y="cnt", alpha=0.25, ax=ax)
    ax.set_title("Rentals vs windspeed")
    save(fig, "08_windspeed_scatter.png")

    num_cols = ["temp", "atemp", "hum", "windspeed", "casual", "registered", "cnt"]
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation of numeric features with cnt")
    save(fig, "09_correlation_heatmap.png")

    day = df[df["dteday"] == df["dteday"].iloc[0]]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(day["hr"], day["casual"], label="casual", color="tab:orange")
    ax.plot(day["hr"], day["registered"], label="registered", color="tab:green")
    ax.set_title("Casual vs registered usage pattern (one sample day)")
    ax.set_xlabel("hour of day")
    ax.legend()
    save(fig, "10_casual_vs_registered.png")

    fig, ax = plt.subplots(figsize=(12, 5))
    for yr, label in [(0, "2011"), (1, "2012")]:
        sub = df[df["yr"] == yr].groupby("mnth")["cnt"].mean()
        ax.plot(sub.index, sub.values, marker="o", label=label)
    ax.set_xticks(range(1, 13))
    ax.set_title("Monthly trend per year")
    ax.legend()
    save(fig, "11_monthly_by_year.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    for wd, label, color in [(1, "weekday", "tab:blue"), (0, "non-working day", "tab:red")]:
        prof = (
            df[df["workingday"] == wd]
            .groupby("hr")["cnt"]
            .mean()
            .reindex(range(24))
        )
        ax.plot(prof.index, prof.values, marker="o", label=label, color=color)
    for h in (7, 8, 9, 17, 18, 19):
        ax.axvline(h, color="grey", ls=":", lw=1)
    ax.set_title("Hourly profile: working vs non-working days (reference paper peak-hour idea)")
    ax.set_xlabel("hour of day")
    ax.legend()
    save(fig, "12_workingday_hourly_profile.png")

    return paths


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe additions; follows the reference paper featurization ideas."""
    out = df.copy()
    out["season_label"] = out["season"].map(SEASON_MAP)
    out["weathersit_label"] = out["weathersit"].map(WEATHER_MAP)

    out["hour_sin"] = np.sin(2 * np.pi * out["hr"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hr"] / 24.0)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["weekday"] / 7.0)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["weekday"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["mnth"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["mnth"] - 1) / 12.0)
    out["is_rush_hour"] = out["hr"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    out["is_weekend"] = (out["weekday"] >= 5).astype(int)

    # Reference paper: 'weekday peak hours are 7-9am and 5-7pm, weekend peak
    # hours are 10am-6pm.' Encode that as a 3-way categorical flag.
    def peak_bucket(row) -> str:
        if row["workingday"] == 1:
            if row["hr"] in (7, 8, 9, 17, 18, 19):
                return "weekday_peak"
            return "weekday_off"
        if 10 <= row["hr"] <= 18:
            return "weekend_peak"
        return "weekend_off"

    out["peak_time"] = out.apply(peak_bucket, axis=1)
    out["temp_bucket"] = pd.qcut(out["atemp"], q=4, labels=["cold", "mild", "warm", "hot"], duplicates="drop").astype(str)
    return out


def build_pipeline() -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ALL_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ALL_CATEGORICAL,
            ),
        ]
    )
    return Pipeline(steps=[("prep", prep), ("reg", LinearRegression())])


ALL_NUMERIC = [
    "temp",
    "atemp",
    "hum",
    "windspeed",
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "is_rush_hour",
    "is_weekend",
]
ALL_CATEGORICAL = ["season_label", "weathersit_label", "peak_time", "temp_bucket", "yr", "holiday", "workingday"]


def temporal_split(df: pd.DataFrame, test_size: float = 0.2):
    split_day = df["dteday"].quantile(1 - test_size)
    mask_test = df["dteday"] > split_day
    return df[~mask_test], df[mask_test], split_day


def evaluate(model, X_te: pd.DataFrame, y_te: pd.Series) -> dict[str, float]:
    pred = np.clip(model.predict(X_te), 0, None)
    return score_report(y_te, pred)


def main() -> None:
    raw = load_raw()
    data_understanding(raw)
    relationship_eda(raw)
    make_figures(raw)

    df = add_engineered_features(raw)
    train, test, split_day = temporal_split(df)
    print(f"Temporal split boundary: {split_day.date()}")

    X_tr, y_tr = df_feature_matrix(train)
    X_te, y_te = df_feature_matrix(test)

    lin_pipe = build_pipeline()
    lin_pipe.fit(X_tr, y_tr)
    lin_metrics = evaluate(lin_pipe, X_te, y_te)
    print("Linear Regression :", lin_metrics)

    rf_pipe = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", ALL_NUMERIC),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ALL_CATEGORICAL,
                        ),
                    ]
                ),
            ),
            ("reg", RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ]
    )
    param_grid = {
        "reg__n_estimators": [200, 400],
        "reg__max_depth": [12, 18, None],
        "reg__min_samples_leaf": [1, 3, 5],
    }
    from sklearn.model_selection import GridSearchCV

    gs = GridSearchCV(
        rf_pipe,
        param_grid,
        cv=TimeSeriesSplit(n_splits=4),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)
    print("Best RF params:", gs.best_params_)
    rf_metrics = evaluate(gs.best_estimator_, X_te, y_te)
    print("Random Forest     :", rf_metrics)

    (MODELS_DIR / "baseline_metrics.json").write_text(
        json.dumps(
            {
                "linear": lin_metrics,
                "random_forest": rf_metrics,
                "best": "random_forest"
                if rf_metrics["rmse"] < lin_metrics["rmse"]
                else "linear",
                "split_date": str(split_day.date()),
            },
            indent=2,
        )
    )
    print("Saved metrics ->", MODELS_DIR / "baseline_metrics.json")


def df_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = ALL_NUMERIC + ALL_CATEGORICAL
    return df[feature_cols], df["cnt"]


if __name__ == "__main__":
    main()
