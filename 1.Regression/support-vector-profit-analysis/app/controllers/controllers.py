"""Controllers for SVR project — multi-model selection."""

import logging
from app.services.training_service import TrainingService
from app.services.prediction_service import PredictionService
from app.schemas.schemas import ProfitPredictionRequest

logger = logging.getLogger(__name__)


class TrainingController:
    def __init__(self):
        self._service = TrainingService()

    def run_training(self):
        result = self._service.train()
        return {
            "status": "success",
            "message": f"Best model selected and saved: {result['best_model']}",
            "best_model": result["best_model"],
            "best_metrics": result["best_metrics"],
            "selection_reason": result["selection_reason"],
            "comparison_table": result["comparison_table"],
        }


class PredictionController:
    def __init__(self):
        self._service = PredictionService()

    def predict(self, request: ProfitPredictionRequest):
        profit = self._service.predict(
            request.rd_spend, request.administration, request.marketing_spend, request.state
        )
        return {
            "rd_spend": request.rd_spend, "administration": request.administration,
            "marketing_spend": request.marketing_spend, "state": request.state,
            "predicted_profit": round(profit, 2), "currency": "USD",
        }
