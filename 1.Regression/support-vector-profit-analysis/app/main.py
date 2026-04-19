"""
SVM Regression — AI in Business Intelligence API
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
    logger.info("🚀  SVM Regression API starting up …")
    yield
    logger.info("🛑  SVM Regression API shutting down …")


app = FastAPI(
    title="Business Intelligence Profit Prediction — SVM Regression API",
    description="Predicts startup profit using Support Vector Regression on R&D Spend, Administration, Marketing Spend, and State.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(training_router, prefix="/train", tags=["Training"])
app.include_router(prediction_router, prefix="/predict", tags=["Prediction"])


@app.get("/", tags=["Root"])
def root():
    return {"project": "Business Intelligence Profit Prediction", "algorithm": "Support Vector Regression", "docs": "/docs"}
