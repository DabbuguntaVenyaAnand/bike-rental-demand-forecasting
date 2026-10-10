"""Model pipelines and TimeSeriesSplit hyperparameter searches."""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation import RMSLE_SCORER
from model_config import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)


def _one_hot_encoder():
    """Support both newer and older scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Build the preprocessing pipeline used before each model."""
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", _one_hot_encoder(), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def _grid_search(pipeline, param_grid, cv):
    """Create temporal GridSearchCV using RMSLE as the scoring metric."""
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=RMSLE_SCORER,
        n_jobs=-1,
        refit=True,
    )


def build_model_searches():
    """
    Build all regression models used in the final comparison.

    Models:
    - Linear Regression: simple baseline
    - KNN Regression
    - Random Forest Regression
    - Gradient Boosting Regression

    All models use the same TimeSeriesSplit strategy so the comparison
    remains consistent and respects the temporal nature of the dataset.
    """
    cv = TimeSeriesSplit(n_splits=4)

    linear_regression = Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=True)),
            ("reg", LinearRegression()),
        ]
    )

    knn = Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=True)),
            ("reg", KNeighborsRegressor()),
        ]
    )

    random_forest = Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=False)),
            (
                "reg",
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    gradient_boosting = Pipeline(
        [
            ("prep", make_preprocessor(scale_numeric=False)),
            ("reg", GradientBoostingRegressor(random_state=RANDOM_STATE)),
        ]
    )

    return {
        "Linear Regression": _grid_search(
            linear_regression,
            {},
            cv,
        ),
        "KNN Regression": _grid_search(
            knn,
            {
                "reg__n_neighbors": [5, 10, 20, 30],
                "reg__weights": ["uniform", "distance"],
                "reg__p": [1, 2],
            },
            cv,
        ),
        "Random Forest": _grid_search(
            random_forest,
            {
                "reg__n_estimators": [300, 400],
                "reg__max_depth": [18, None],
                "reg__min_samples_leaf": [1, 3],
            },
            cv,
        ),
        "Gradient Boosting": _grid_search(
            gradient_boosting,
            {
                "reg__n_estimators": [100, 200],
                "reg__learning_rate": [0.05, 0.10],
                "reg__max_depth": [2, 3],
            },
            cv,
        ),
    }
