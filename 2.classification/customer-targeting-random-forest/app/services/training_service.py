"""
Training Service — Social Network Ads Purchase Prediction (RF-CLF base)
Multi-Model Selection: Trains ALL classifiers, compares metrics, auto-selects best.

Models evaluated:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Support Vector Classifier (SVC)

Selection metric: F1 Score
"""

import logging
import pathlib
from typing import Any, Dict, List

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.utils.config import settings

logger = logging.getLogger(__name__)

RAW_DATA_PATH = pathlib.Path(settings.raw_data_path)
MODEL_PATH = pathlib.Path(settings.model_path)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = ["Age", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Gender"]
TARGET = "Purchased"

CANDIDATE_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=500, random_state=settings.random_state),
    "DecisionTreeClassifier": DecisionTreeClassifier(criterion="entropy", random_state=settings.random_state),
    "RandomForestClassifier": RandomForestClassifier(
        n_estimators=settings.n_estimators, criterion="entropy",
        max_depth=settings.max_depth, random_state=settings.random_state
    ),
    "SVC": SVC(kernel="rbf", probability=True, random_state=settings.random_state),
}


class TrainingService:
    """Multi-model classifier selection — RF project variant."""

    def train(self) -> Dict[str, Any]:
        df = self._load_data()
        X, y = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=settings.test_size, random_state=settings.random_state
        )
        preprocessor = ColumnTransformer(transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
        ])

        comparison_table: List[Dict[str, Any]] = []
        best_pipeline, best_model_name, best_f1 = None, "", float("-inf")

        for name, estimator in CANDIDATE_MODELS.items():
            pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
            pipeline.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, pipeline.predict(X_train))
            y_pred    = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec  = recall_score(y_test, y_pred, zero_division=0)
            f1   = f1_score(y_test, y_pred, zero_division=0)

            row = {
                "model": name, "train_accuracy": round(train_acc, 4),
                "accuracy": round(acc, 4), "precision": round(prec, 4),
                "recall": round(rec, 4), "f1_score": round(f1, 4),
                "overfit_gap": round(train_acc - acc, 4),
            }
            comparison_table.append(row)
            logger.info("%s | acc=%.4f | F1=%.4f", name, acc, f1)

            if f1 > best_f1:
                best_f1, best_pipeline, best_model_name = f1, pipeline, name

        joblib.dump(best_pipeline, MODEL_PATH)
        logger.info("✅ Best model: %s (F1=%.4f) saved → %s", best_model_name, best_f1, MODEL_PATH)
        best_row = next(r for r in comparison_table if r["model"] == best_model_name)
        return {
            "best_model": best_model_name,
            "best_metrics": {
                "accuracy": best_row["accuracy"], "precision": best_row["precision"],
                "recall": best_row["recall"], "f1_score": best_row["f1_score"],
            },
            "selection_reason": (
                f"{best_model_name} achieved the highest F1 score ({round(best_f1, 4)}) "
                "balancing precision and recall on the hold-out set."
            ),
            "note": "Model selection is data-dependent. Results may vary with different data distributions.",
            "comparison_table": comparison_table,
        }

    def _load_data(self) -> pd.DataFrame:
        if not RAW_DATA_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {RAW_DATA_PATH}")
        df = pd.read_csv(RAW_DATA_PATH).dropna()
        if "User ID" in df.columns:
            df = df.drop("User ID", axis=1)
        return df
