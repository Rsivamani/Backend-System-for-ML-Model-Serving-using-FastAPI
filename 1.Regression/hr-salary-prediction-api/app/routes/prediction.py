"""Prediction route."""

import logging

from fastapi import APIRouter, HTTPException

from app.controllers.prediction_controller import PredictionController
from app.schemas.request_schema import SalaryPredictionRequest
from app.schemas.response_schema import SalaryPredictionResponse

logger = logging.getLogger(__name__)
router = APIRouter()
_ctrl = PredictionController()


@router.post("/", response_model=SalaryPredictionResponse, summary="Predict Salary")
def predict_salary(payload: SalaryPredictionRequest):
    """
    Accepts years of experience and returns the predicted annual salary.
    """
    try:
        return _ctrl.predict(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
