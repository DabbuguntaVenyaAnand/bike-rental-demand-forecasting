"""Regression metrics and result visualizations."""

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

from model_config import REPORTS_DIR


def rmsle_value(y_true, y_pred) -> float:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


RMSLE_SCORER = make_scorer(rmsle_value, greater_is_better=False)


def score_report(y_true, y_pred) -> dict:
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)

    return {
        "rmsle": rmsle_value(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_comparison_plot(results: pd.DataFrame) -> None:
    ordered = results.sort_values("test_rmsle")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(ordered["model"], ordered["test_rmsle"])

    ax.set_title("Final Model Comparison — RMSLE")
    ax.set_ylabel("RMSLE (lower is better)")
    ax.tick_params(axis="x", rotation=15)

    for bar, value in zip(bars, ordered["test_rmsle"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "15_final_model_comparison.png", dpi=140)
    plt.close(fig)


def save_prediction_plot(y_test, pred, model_name: str) -> None:
    n = min(len(y_test), 24 * 14)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test.iloc[:n].to_numpy(), label="actual", lw=1.4)
    ax.plot(pred[:n], label=f"predicted: {model_name}", lw=1.1)

    ax.set_title("Best Model — Actual vs Predicted (First 2 Test Weeks)")
    ax.set_xlabel("hour index")
    ax.set_ylabel("bike rentals (cnt)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "16_best_model_pred_vs_actual.png", dpi=140)
    plt.close(fig)
