"""Routes for SVM Classification API."""

from fastapi import APIRouter, HTTPException
from app.controllers.controllers import TrainingController, PredictionController
from app.schemas.schemas import PurchasePredictionRequest, PurchasePredictionResponse

health_router = APIRouter()
training_router = APIRouter()
prediction_router = APIRouter()

_train_ctrl = TrainingController()
_pred_ctrl = PredictionController()


@health_router.get("/", summary="Health Check")
def health():
    return {"status": "ok", "service": "Social Network Ads — SVM Classification API"}


@training_router.post("/", summary="Train SVM Classifier")
def train():
    try:
        return _train_ctrl.run_training()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@prediction_router.post("/", response_model=PurchasePredictionResponse, summary="Predict Purchase (SVM)")
def predict(payload: PurchasePredictionRequest):
    try:
        return _pred_ctrl.predict(payload)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
