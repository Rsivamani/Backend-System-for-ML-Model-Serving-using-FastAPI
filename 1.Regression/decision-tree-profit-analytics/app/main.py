"""
Decision Tree Regression — HR Salary Prediction (Multi-feature)
FastAPI entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import prediction_router, training_router, health_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀  Decision Tree Regression API starting up …")
    yield
    logger.info("🛑  Decision Tree Regression API shutting down …")


app = FastAPI(
    title="HR Salary Prediction — Decision Tree Regression API",
    description="Predicts startup profit from multiple features using Decision Tree Regression.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(training_router, prefix="/train", tags=["Training"])
app.include_router(prediction_router, prefix="/predict", tags=["Prediction"])


@app.get("/", tags=["Root"])
def root():
    return {"project": "HR Salary - Decision Tree Regression", "algorithm": "DecisionTreeRegressor", "docs": "/docs"}
