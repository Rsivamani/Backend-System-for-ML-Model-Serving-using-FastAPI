"""
Training Service — Business Intelligence Profit Prediction (SVR base)
Multi-Model Selection: Identical logic to MLR project.
Compares: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, SVR
"""

import logging
import pathlib
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from app.utils.config import settings

logger = logging.getLogger(__name__)

RAW_DATA_PATH = pathlib.Path(settings.raw_data_path)
MODEL_PATH = pathlib.Path(settings.model_path)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]
TARGET = "Profit"

CANDIDATE_MODELS = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(random_state=settings.random_state),
    "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=settings.random_state),
    "SVR": SVR(kernel="rbf", C=1e5, epsilon=0.1),
}


class TrainingService:
    """Multi-model regression training with auto-selection for SVR project."""

    def train(self) -> Dict[str, Any]:
        df = self._load_data()
        X, y = self._preprocess(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.test_size, random_state=settings.random_state
        )
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
        ])

        comparison_table: List[Dict[str, Any]] = []
        best_pipeline, best_model_name, best_r2 = None, "", float("-inf")

        for name, estimator in CANDIDATE_MODELS.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            pipeline.fit(X_train, y_train)

            train_r2 = r2_score(y_train, pipeline.predict(X_train))
            y_pred   = pipeline.predict(X_test)
            test_r2  = r2_score(y_test, y_pred)
            mae      = mean_absolute_error(y_test, y_pred)
            mse      = mean_squared_error(y_test, y_pred)

            row = {
                "model": name,
                "train_r2": round(train_r2, 4), "test_r2": round(test_r2, 4),
                "mae": round(mae, 2), "mse": round(mse, 2),
                "rmse": round(mse ** 0.5, 2), "overfit_gap": round(train_r2 - test_r2, 4),
            }
            comparison_table.append(row)
            logger.info("%s | R²=%.4f | MAE=%.2f", name, test_r2, mae)

            if test_r2 > best_r2:
                best_r2, best_pipeline, best_model_name = test_r2, pipeline, name

        joblib.dump(best_pipeline, MODEL_PATH)
        logger.info("✅ Best model: %s saved → %s", best_model_name, MODEL_PATH)
        best_row = next(r for r in comparison_table if r["model"] == best_model_name)
        return {
            "best_model": best_model_name,
            "best_metrics": {"test_r2": best_row["test_r2"], "mae": best_row["mae"], "rmse": best_row["rmse"]},
            "selection_reason": f"{best_model_name} yielded highest test R² ({round(best_r2, 4)}) with best generalisation.",
            "comparison_table": comparison_table,
        }

    def _load_data(self) -> pd.DataFrame:
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")
        return pd.read_csv(RAW_DATA_PATH).dropna()

    def _preprocess(self, df: pd.DataFrame):
        return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]
