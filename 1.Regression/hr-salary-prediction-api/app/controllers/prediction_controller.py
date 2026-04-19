"""Prediction controller."""

import logging
from typing import Any, Dict

from app.schemas.request_schema import SalaryPredictionRequest
from app.services.prediction_service import PredictionService

logger = logging.getLogger(__name__)


class PredictionController:
    """Bridges the /predict route and the PredictionService."""

    def __init__(self):
        self._service = PredictionService()

    def predict(self, request: SalaryPredictionRequest) -> Dict[str, Any]:
        logger.info(
            "PredictionController: years_experience=%.2f", request.years_experience
        )
        predicted_salary = self._service.predict(request.years_experience)
        return {
            "years_experience": request.years_experience,
            "predicted_salary": round(predicted_salary, 2),
            "currency": "USD",
        }
