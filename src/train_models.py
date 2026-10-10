"""Train, compare and select the final bike-rental demand model.

Run from the repository root:
    python src/train_models.py
"""

import json

import joblib
import numpy as np
import pandas as pd

from evaluation import (
    save_comparison_plot,
    save_prediction_plot,
    score_report,
)
from feature_engineering import (
    feature_matrix,
    load_dataset,
    temporal_split,
    validate_design,
)
from model_config import ALL_FEATURES, MODELS_DIR
from model_training import build_model_searches


def main() -> None:
    df = load_dataset()

    train, test, split_day = temporal_split(df)

    X_train, y_train = feature_matrix(train)
    X_test, y_test = feature_matrix(test)

    validate_design(train, X_train)
    validate_design(test, X_test)

    print(f"Rows: {len(df):,}")
    print(f"Train: {len(train):,} | Test: {len(test):,}")
    print(f"Split boundary: {split_day.date()}")
    print("Leakage columns are excluded from the feature matrix.")

    searches = build_model_searches()
    fitted = {}
    rows = []

    for model_name, search in searches.items():
        print(f"\n=== {model_name} ===")

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        test_pred = np.clip(best_model.predict(X_test), 0, None)

        test_metrics = score_report(y_test, test_pred)
        cv_rmsle = float(-search.best_score_)

        print("Best params:", search.best_params_)
        print(f"CV RMSLE:   {cv_rmsle:.4f}")
        print(f"Test RMSLE: {test_metrics['rmsle']:.4f}")

        fitted[model_name] = {
            "search": search,
            "model": best_model,
            "pred": test_pred,
            "metrics": test_metrics,
            "cv_rmsle": cv_rmsle,
        }

        rows.append(
            {
                "model": model_name,
                "cv_rmsle": cv_rmsle,
                "test_rmsle": test_metrics["rmsle"],
                "rmse": test_metrics["rmse"],
                "mae": test_metrics["mae"],
                "r2": test_metrics["r2"],
            }
        )

    results = (
        pd.DataFrame(rows)
        .sort_values("test_rmsle")
        .reset_index(drop=True)
    )

    print("\n=== FINAL HOLDOUT COMPARISON ===")
    print(results.round(4).to_string(index=False))

    # Choose the production model using training-period CV only.
    best_name = min(fitted, key=lambda name: fitted[name]["cv_rmsle"])
    best = fitted[best_name]

    print(f"\nSelected production model by CV RMSLE: {best_name}")
    print("Holdout metrics:", best["metrics"])

    joblib.dump(best["model"], MODELS_DIR / "production_model.pkl")
    results.to_csv(MODELS_DIR / "model_comparison.csv", index=False)

    metrics_payload = {
        "selection_rule": "lowest TimeSeriesSplit CV RMSLE on training period",
        "selected_model": best_name,
        "split_boundary": str(split_day.date()),
        "results": rows,
    }

    (MODELS_DIR / "model_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2)
    )

    diagnostics = {
        "selected_model": best_name,
        "selected_params": best["search"].best_params_,
        "features": ALL_FEATURES,
        "excluded_leakage_columns": ["casual", "registered", "cnt", "instant"],
        "cv": "TimeSeriesSplit(n_splits=4)",
        "primary_metric": "RMSLE",
        "notes": [
            "All final models use the same feature set for fair comparison.",
            "Quantile temperature buckets are omitted to avoid split inconsistency.",
            "Hour, weekday and month use categorical/cyclic representations.",
        ],
    }

    (MODELS_DIR / "model_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2)
    )

    save_comparison_plot(results)
    save_prediction_plot(y_test, best["pred"], best_name)

    print("\nSaved final model, metrics, diagnostics and plots.")


if __name__ == "__main__":
    main()
