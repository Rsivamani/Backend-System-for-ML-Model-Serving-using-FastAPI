"""
Social Network Ads — Decision Tree Classification API
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
    logger.info("🚀  DT Classification API starting up …")
    yield
    logger.info("🛑  DT Classification API shutting down …")


app = FastAPI(
    title="Social Network Ads — Purchase Prediction (Decision Tree Classifier)",
    description="Predicts whether a user will purchase a product based on Age, Estimated Salary, and Gender.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(training_router, prefix="/train", tags=["Training"])
app.include_router(prediction_router, prefix="/predict", tags=["Prediction"])


@app.get("/", tags=["Root"])
def root():
    return {"project": "Social Network Ads Purchase Prediction", "algorithm": "Decision Tree Classifier", "docs": "/docs"}
