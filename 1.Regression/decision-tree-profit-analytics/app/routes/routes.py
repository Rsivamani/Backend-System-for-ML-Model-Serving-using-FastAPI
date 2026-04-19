"""Routes for DT Regression Profit Prediction API."""

from fastapi import APIRouter, HTTPException
from app.controllers.controllers import TrainingController, PredictionController
from app.schemas.schemas import ProfitPredictionRequest, ProfitPredictionResponse

health_router = APIRouter()
training_router = APIRouter()
prediction_router = APIRouter()

_train_ctrl = TrainingController()
_pred_ctrl = PredictionController()


@health_router.get("/", summary="Health Check")
def health():
    return {"status": "ok", "service": "Decision Tree Regression API"}


@training_router.post("/", summary="Train DT Regressor")
def train():
    try:
        return _train_ctrl.run_training()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@prediction_router.post("/", response_model=ProfitPredictionResponse, summary="Predict Profit (DT)")
def predict(payload: ProfitPredictionRequest):
    try:
        return _pred_ctrl.predict(payload)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
