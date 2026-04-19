"""Training route — triggers model re-training."""

import logging

from fastapi import APIRouter, HTTPException

from app.controllers.training_controller import TrainingController

logger = logging.getLogger(__name__)
router = APIRouter()
_ctrl = TrainingController()


@router.post("/", summary="Train the SLR model")
def train_model():
    """
    Loads the raw dataset, trains a Simple Linear Regression model,
    persists it to disk and returns evaluation metrics.
    """
    try:
        result = _ctrl.run_training()
        return result
    except FileNotFoundError as exc:
        logger.error("Dataset not found: %s", exc)
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Training failed")
        raise HTTPException(status_code=500, detail=str(exc))
