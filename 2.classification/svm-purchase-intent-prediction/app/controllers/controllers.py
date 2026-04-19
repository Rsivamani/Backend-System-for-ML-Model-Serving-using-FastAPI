"""Controllers for SVM Classification project — multi-model selection."""

import logging
from app.services.training_service import TrainingService
from app.services.prediction_service import PredictionService
from app.schemas.schemas import PurchasePredictionRequest

logger = logging.getLogger(__name__)


class TrainingController:
    def __init__(self):
        self._service = TrainingService()

    def run_training(self):
        result = self._service.train()
        return {
            "status": "success",
            "message": f"Best classifier selected and saved: {result['best_model']}",
            "best_model": result["best_model"],
            "best_metrics": result["best_metrics"],
            "selection_reason": result["selection_reason"],
            "note": result.get("note", ""),
            "comparison_table": result["comparison_table"],
        }


class PredictionController:
    def __init__(self):
        self._service = PredictionService()

    def predict(self, request: PurchasePredictionRequest):
        purchased, purchase_probability, confidence = self._service.predict(
            request.age, request.estimated_salary, request.gender
        )
        return {
            "age": request.age, "estimated_salary": request.estimated_salary,
            "gender": request.gender, "purchased": purchased,
            "purchase_probability": purchase_probability, "confidence": confidence,
            "prediction_label": "Will Purchase" if purchased == 1 else "Will NOT Purchase",
        }
