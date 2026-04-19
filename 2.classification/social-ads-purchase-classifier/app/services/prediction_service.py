"""Prediction Service for DT Classification."""

import logging
import pathlib

import joblib
import numpy as np
import pandas as pd

from app.utils.config import settings

logger = logging.getLogger(__name__)
MODEL_PATH = pathlib.Path(settings.model_path)
VALID_GENDERS = {"Male", "Female"}
NUMERIC_FEATURES = ["Age", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Gender"]


class PredictionService:
    def __init__(self):
        self._pipeline = None

    def predict(self, age: int, estimated_salary: float, gender: str):
        if gender not in VALID_GENDERS:
            raise ValueError(f"gender must be one of {VALID_GENDERS}. Got: '{gender}'")
        pipeline = self._get_pipeline()
        X = pd.DataFrame([{"Age": age, "EstimatedSalary": estimated_salary, "Gender": gender}])
        predicted_class = int(pipeline.predict(X)[0])
        probabilities = pipeline.predict_proba(X)[0]
        confidence = round(float(np.max(probabilities)), 4)
        purchase_probability = round(float(probabilities[1]), 4)
        logger.info("DT CLF → age=%d salary=%.2f gender=%s → purchased=%d (conf=%.4f)",
                    age, estimated_salary, gender, predicted_class, confidence)
        return predicted_class, purchase_probability, confidence

    def _get_pipeline(self):
        if self._pipeline is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Call POST /train first.")
            self._pipeline = joblib.load(MODEL_PATH)
        return self._pipeline
