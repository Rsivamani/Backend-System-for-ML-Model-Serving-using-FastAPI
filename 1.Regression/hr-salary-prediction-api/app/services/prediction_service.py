"""Prediction Service — loads persisted model and returns salary estimate."""

import logging
import pathlib

import joblib
import numpy as np

from app.utils.config import settings

logger = logging.getLogger(__name__)

MODEL_PATH = pathlib.Path(settings.model_path)


class PredictionService:
    """Handles model loading and inference."""

    def __init__(self):
        self._pipeline = None

    # ── Public API ────────────────────────────────────────────────────────────
    def predict(self, years_experience: float) -> float:
        pipeline = self._get_pipeline()
        if years_experience < 0:
            raise ValueError("years_experience must be non-negative.")
        X = np.array([[years_experience]])
        predicted = pipeline.predict(X)[0]
        logger.info(
            "Prediction | years_experience=%.2f → salary=%.2f",
            years_experience,
            predicted,
        )
        return float(predicted)

    # ── Private helpers ───────────────────────────────────────────────────────
    def _get_pipeline(self):
        if self._pipeline is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model not found at {MODEL_PATH}. Call POST /train first."
                )
            self._pipeline = joblib.load(MODEL_PATH)
            logger.info("Model loaded from %s", MODEL_PATH)
        return self._pipeline
