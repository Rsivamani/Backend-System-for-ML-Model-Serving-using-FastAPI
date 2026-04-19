from app.routes.health import router as health_router
from app.routes.training import router as training_router
from app.routes.prediction import router as prediction_router

__all__ = ["health_router", "training_router", "prediction_router"]
