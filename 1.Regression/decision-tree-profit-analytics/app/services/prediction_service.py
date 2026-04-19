"""Prediction Service for Decision Tree Regression."""

import logging
import pathlib

import joblib
import pandas as pd

from app.utils.config import settings

logger = logging.getLogger(__name__)
MODEL_PATH = pathlib.Path(settings.model_path)
VALID_STATES = {"New York", "California", "Florida"}
NUMERIC_FEATURES = ["R&D Spend", "Administration", "Marketing Spend"]
CATEGORICAL_FEATURES = ["State"]


class PredictionService:
    def __init__(self):
        self._pipeline = None

    def predict(self, rd_spend: float, administration: float, marketing_spend: float, state: str) -> float:
        if state not in VALID_STATES:
            raise ValueError(f"state must be one of {VALID_STATES}. Got: '{state}'")
        pipeline = self._get_pipeline()
        X = pd.DataFrame([{"R&D Spend": rd_spend, "Administration": administration, "Marketing Spend": marketing_spend, "State": state}])
        predicted = pipeline.predict(X)[0]
        logger.info("DT Regression → profit=%.2f", predicted)
        return float(predicted)

    def _get_pipeline(self):
        if self._pipeline is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Call POST /train first.")
            self._pipeline = joblib.load(MODEL_PATH)
        return self._pipeline
