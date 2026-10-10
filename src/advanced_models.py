"""Advanced models for the team report (runs on the SAME features and temporal
split as src/preprocess_and_baseline.py).

Trains Gradient Boosting, RBF-SVR and a 50/50 GBM+RF blend, reports
RMSLE/RMSE/MAE/R2, and writes models/advanced_metrics.json.

Run (.venv, from src/):
    .venv/Scripts/python src/advanced_models.py
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from preprocess_and_baseline import (
    ALL_CATEGORICAL,
    ALL_NUMERIC,
    MODELS_DIR,
    RANDOM_STATE,
    add_engineered_features,
    feature_matrix,
    fit_temp_bucket_edges,
    load_dataset,
    score_report,
    temporal_split,
)


def build_scaled_tree_pipeline(model) -> Pipeline:
    prep = ColumnTransformer(
        transformers=[
            ("num", "passthrough", ALL_NUMERIC),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ALL_CATEGORICAL,
            ),
        ]
    )
    return Pipeline(steps=[("prep", prep), ("reg", model)])


def main() -> None:
    df = load_dataset()
    train, test, _ = temporal_split(df)
    fit_temp_bucket_edges(train)
    X_tr, y_tr = feature_matrix(add_engineered_features(train))
    X_te, y_te = feature_matrix(add_engineered_features(test))

    results = {}

    # --- Gradient Boosting (paper's GBM idea) ---
    t0 = time.time()
    gbr = build_scaled_tree_pipeline(
        GradientBoostingRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.1, subsample=0.8,
            random_state=RANDOM_STATE,
        )
    )
    gbr.fit(X_tr, y_tr)
    results["gradient_boosting"] = score_report(y_te, gbr.predict(X_te))
    print(f"GBM done in {time.time()-t0:.0f}s:", results["gradient_boosting"])

    # --- Random Forest (fresh copy for the blend) ---
    rf = build_scaled_tree_pipeline(
        RandomForestRegressor(
            n_estimators=400, max_depth=None, min_samples_leaf=1,
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    )
    rf.fit(X_tr, y_tr)
    results["random_forest"] = score_report(y_te, rf.predict(X_te))
    print("RF done:", results["random_forest"])

    # --- Simple 50/50 blend of GBM and RF ---
    blend = (np.clip(gbr.predict(X_te), 0, None) + np.clip(rf.predict(X_te), 0, None)) / 2
    results["blend_gbm_rf"] = score_report(y_te, blend)
    print("Blend done:", results["blend_gbm_rf"])

    # --- RBF-SVR (paper's SVR idea); slowest, so last ---
    t0 = time.time()
    print("SVR training (may take a few minutes)...")
    svr = Pipeline(
        steps=[
            (
                "prep",
                ColumnTransformer(
                    transformers=[
                        ("num", StandardScaler(), ALL_NUMERIC),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                            ALL_CATEGORICAL,
                        ),
                    ]
                ),
            ),
            ("reg", SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")),
        ]
    )
    svr.fit(X_tr, y_tr)
    results["svr_rbf"] = score_report(y_te, svr.predict(X_te))
    print(f"SVR done in {time.time()-t0:.0f}s:", results["svr_rbf"])

    best = min(results, key=lambda k: results[k]["rmsle"])
    results["best_by_rmsle"] = best
    (MODELS_DIR / "advanced_metrics.json").write_text(json.dumps(results, indent=2))
    print("Best by RMSLE:", best)
    print("Saved ->", MODELS_DIR / "advanced_metrics.json")


if __name__ == "__main__":
    main()
