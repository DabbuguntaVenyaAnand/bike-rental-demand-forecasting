"""Bike Rental Demand Forecasting - baseline preprocessing and modeling.

Preprocessing, feature engineering, temporal split and baseline models
(Linear Regression + Random Forest) with leakage-safe inputs only.

Featurization follows the reference paper (Du, He, Zhechev - "Forecasting Bike
Rental Demand"): discretized temperature buckets, weekday/weekend peak-hour
buckets, dropping raw temp in favor of atemp (collinearity), keeping the month
variable, and reporting the competition metric RMSLE alongside RMSE/MAE/R2.

Writes:
    models/baseline_model.pkl           fitted best baseline pipeline
    models/baseline_metrics.json        test-set metrics (best model)
    models/baseline_diagnostics.json    fitted params + engineered columns
    reports/13_baseline_pred_vs_actual.png
    reports/14_baseline_residuals.png

Run (.venv):
    .venv/Scripts/python src/preprocess_and_baseline.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sns.set_theme(style="whitegrid", context="notebook")
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
WEEKEND_PEAK_HOURS = tuple(range(10, 19))  # 10am - 6pm inclusive

TEMP_BUCKET_LABELS = ["cold", "mild", "warm", "hot"]
TEMP_BUCKET_EDGES: np.ndarray | None = None

NUMERIC_FEATURES = ["atemp", "hum", "windspeed"]
ENGINEERED_NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "dayofweek_sin",
    "dayofweek_cos",
    "month_sin",
    "month_cos",
    "is_rush_hour",
    "is_weekend",
]
ALL_NUMERIC = NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES
ALL_CATEGORICAL = [
    "peak_time",
    "temp_bucket",
    "season_label",
    "weathersit_label",
    "mnth",
    "yr",
    "holiday",
    "workingday",
]


def compute_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSLE - the Kaggle competition metric used in the reference paper."""
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2)))


RMSLE_SCORER = make_scorer(compute_rmsle, greater_is_better=False)


def score_report(y_true, y_pred) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "rmsle": compute_rmsle(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["dteday"])
    return df.dropna().reset_index(drop=True)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leakage-safe additions; follows the reference paper featurization ideas.

    NOTE: 'casual' + 'registered' == 'cnt', so those columns stay excluded.
    """
    out = df.copy()
    out["season_label"] = out["season"].map(SEASON_MAP)
    out["weathersit_label"] = out["weathersit"].map(WEATHER_MAP)

    out["hour_sin"] = np.sin(2 * np.pi * out["hr"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hr"] / 24.0)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["weekday"] / 7.0)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["weekday"] / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * (out["mnth"] - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (out["mnth"] - 1) / 12.0)
    out["is_rush_hour"] = (
        (out["workingday"] == 1) & out["hr"].isin(WEEKDAY_PEAK_HOURS)
    ).astype(int)
    # NOTE: the UCI dataset codes weekday as Sunday=0 ... Saturday=6,
    # so the weekend days are 0 and 6 (NOT a >= 5 comparison).
    out["is_weekend"] = out["weekday"].isin([0, 6]).astype(int)

    def peak_bucket(row) -> str:
        if row["workingday"] == 1:
            if row["hr"] in WEEKDAY_PEAK_HOURS:
                return "weekday_peak"
            return "weekday_off"
        if row["hr"] in WEEKEND_PEAK_HOURS:
            return "weekend_peak"
        return "weekend_off"

    out["peak_time"] = out.apply(peak_bucket, axis=1)

    if TEMP_BUCKET_EDGES is None:
        raise RuntimeError(
            "Call fit_temp_bucket_edges(train_df) before add_engineered_features."
        )
    out["temp_bucket"] = assign_temp_bucket(out["atemp"], TEMP_BUCKET_EDGES)
    return out


def fit_temp_bucket_edges(train_df: pd.DataFrame) -> np.ndarray:
    """Fit the atemp quantile bucket edges on TRAINING data only and store them
    in the module-level TEMP_BUCKET_EDGES, so the same boundaries are reused for
    test data and for the demo app (no leakage, no inconsistent bucketing)."""
    global TEMP_BUCKET_EDGES
    _, edges = pd.qcut(
        train_df["atemp"], q=4, labels=TEMP_BUCKET_LABELS, retbins=True,
        duplicates="drop",
    )
    TEMP_BUCKET_EDGES = edges
    return edges


def assign_temp_bucket(atemp_norm, edges: np.ndarray):
    """Assign bucket labels using fixed edges. Input must already be in the
    dataset's normalized atemp space. UCI defines atemp using a -16°C to +50°C range, i.e. (T + 16) / 66 for a Celsius value."""
    return pd.cut(
        atemp_norm, bins=edges, labels=TEMP_BUCKET_LABELS, include_lowest=True,
    ).astype(str)


def feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = ALL_NUMERIC + ALL_CATEGORICAL
    return df[feature_cols], df["cnt"]


def temporal_split(df: pd.DataFrame, test_size: float = 0.2):
    split_day = df["dteday"].quantile(1 - test_size)
    mask_test = df["dteday"] > split_day
    return df[~mask_test], df[mask_test], split_day


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


def evaluate(model, X_te, y_te) -> dict:
    return score_report(y_te, model.predict(X_te))


def _save_pred_vs_actual(y_te, preds: dict) -> None:
    n = min(len(y_te), 24 * 14)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_te.values[:n], label="actual", color="black", lw=1.3)
    for name, p in preds.items():
        ax.plot(p[:n], label=f"pred: {name}", lw=1.0, alpha=0.85)
    ax.set_title("Baseline predictions vs actual hourly demand (first 2 test weeks)")
    ax.set_xlabel("hour index (test set)")
    ax.set_ylabel("cnt")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "13_baseline_pred_vs_actual.png", dpi=120)
    plt.close(fig)


def _save_residuals(y_te, preds: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Baseline model residuals and fit quality")
    for ax, (name, p) in zip(axes, preds.items()):
        resid = y_te.values - p
        ax.scatter(p, resid, s=8, alpha=0.25)
        ax.axhline(0, color="red", lw=1.2, ls="--")
        ax.set_title(name)
        ax.set_xlabel("predicted cnt")
        ax.set_ylabel("residual (actual - pred)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "14_baseline_residuals.png", dpi=120)
    plt.close(fig)


def main() -> None:
    df = load_dataset()

    train, test, split_day = temporal_split(df)
    print(
        f"Temporal split -> train rows: {len(train):,} | test rows: {len(test):,}"
    )
    print(f"Split boundary date: {split_day.date()}")

    edges = fit_temp_bucket_edges(train)
    print("Temp bucket edges (train-fitted):", [round(float(e), 4) for e in edges])
    (MODELS_DIR / "temp_bucket_edges.json").write_text(
        json.dumps({"edges": [float(e) for e in edges], "labels": TEMP_BUCKET_LABELS})
    )

    X_tr, y_tr = feature_matrix(add_engineered_features(train))
    X_te, y_te = feature_matrix(add_engineered_features(test))

    # --- Linear Regression baseline ---
    lin_pipe = build_pipeline()
    lin_pipe.fit(X_tr, y_tr)
    lin_metrics = evaluate(lin_pipe, X_te, y_te)
    print("Linear Regression :", lin_metrics)

    # --- Random Forest baseline (small grid) ---
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
    gs = GridSearchCV(
        rf_pipe,
        param_grid,
        cv=TimeSeriesSplit(n_splits=4),
        scoring=RMSLE_SCORER,
        n_jobs=-1,
    )
    gs.fit(X_tr, y_tr)
    print("Best RF params:", gs.best_params_)
    rf_metrics = evaluate(gs.best_estimator_, X_te, y_te)
    print("Random Forest     :", rf_metrics)

    preds_lin = np.clip(lin_pipe.predict(X_te), 0, None)
    preds_rf = np.clip(gs.best_estimator_.predict(X_te), 0, None)
    _save_pred_vs_actual(y_te, {"linear": preds_lin, "random_forest": preds_rf})
    _save_residuals(y_te, {"linear": preds_lin, "random_forest": preds_rf})

    if rf_metrics["rmsle"] <= lin_metrics["rmsle"]:
        best_name = "random_forest"
        best_obj = gs.best_estimator_
        best_metrics = rf_metrics
        best_params = {
            k: (None if v is None else int(v)) for k, v in gs.best_params_.items()
        }
    else:
        best_name = "linear_regression"
        best_obj = lin_pipe
        best_metrics = lin_metrics
        best_params = {"model": "ols"}

    joblib.dump(best_obj, MODELS_DIR / "baseline_model.pkl")
    (MODELS_DIR / "baseline_metrics.json").write_text(
        json.dumps({"best": best_name, **best_metrics}, indent=2)
    )
    (MODELS_DIR / "baseline_diagnostics.json").write_text(
        json.dumps(
            {
                "chosen_model": best_name,
                "best_params": best_params,
                "engineered_features": ENGINEERED_NUMERIC_FEATURES,
                "features": ALL_NUMERIC + ALL_CATEGORICAL,
                "linear": lin_metrics,
                "random_forest": rf_metrics,
                "cv_best_score_rmsle": float(-gs.best_score_),
                "temporal_split_date_threshold": str(split_day.date()),
                "reference_paper": "Du, He, Zhechev - Forecasting Bike Rental Demand",
                "version": "baseline-v3-rmsle-tuned",
            },
            indent=2,
        )
    )
    print("Saved artifacts in", MODELS_DIR)
    print("Saved plots in", REPORTS_DIR)


if __name__ == "__main__":
    main()
