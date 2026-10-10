"""Bike Rental Demand Forecasting - data understanding and EDA.

This script performs dataset inspection and exploratory data analysis only.
Model training is handled by the dedicated training scripts so rerunning EDA
never overwrites model artifacts.

Writes: reports/01..12 png figures.

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

sns.set_theme(style="whitegrid", context="talk")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hour.csv"
FIG_DIR = ROOT / "reports"

FIG_DIR.mkdir(exist_ok=True)

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
    print("\n=== DUPLICATE HOURLY OBSERVATIONS (dteday, hr) ===", df.duplicated(subset=["dteday", "hr"]).sum())
    print("\n=== DATE RANGE ===")
    print(df["dteday"].min(), "->", df["dteday"].max())
    print("\n=== TARGET STATISTICS ===")
    print(df["cnt"].describe().round(2))
    print("Skew:", round(df["cnt"].skew(), 3))
    # Sanity check of the weekday convention (Sunday=0 ... Saturday=6):
    sample = df.groupby(df["dteday"].dt.day_name())["weekday"].median()
    print("\n=== WEEKDAY CODES (median code per calendar day) ===")
    print(sample)


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


def main() -> None:
    raw = load_raw()
    data_understanding(raw)
    relationship_eda(raw)
    make_figures(raw)
    print("EDA finished. (Modeling lives in src/preprocess_and_baseline.py)")


if __name__ == "__main__":
    main()
