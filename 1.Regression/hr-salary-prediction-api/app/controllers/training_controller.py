"""Training controller — SLR project (upgraded for multi-model response)."""

import logging
from typing import Any, Dict
from app.services.training_service import TrainingService

logger = logging.getLogger(__name__)


class TrainingController:
    """Orchestrates model selection pipeline and returns structured comparison results."""

    def __init__(self):
        self._service = TrainingService()

    def run_training(self) -> Dict[str, Any]:
        logger.info("TrainingController: starting multi-model regression pipeline")
        result = self._service.train()
        logger.info(
            "TrainingController: training complete — best model: %s | R²: %s",
            result["best_model"], result["best_metrics"].get("test_r2")
        )
        return {
            "status": "success",
            "message": f"Best model selected: {result['best_model']}",
            "best_model": result["best_model"],
            "best_metrics": result["best_metrics"],
            "selection_reason": result["selection_reason"],
            "comparison_table": result["comparison_table"],
        }
