"""
HR Salary Prediction API — Simple Linear Regression
FastAPI entry point
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import prediction_router, training_router, health_router

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀  HR Salary Prediction API starting up …")
    yield
    logger.info("🛑  HR Salary Prediction API shutting down …")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HR Salary Prediction API",
    description=(
        "Production-ready REST API for predicting employee salaries "
        "using Simple Linear Regression on Years of Experience."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(training_router, prefix="/train", tags=["Training"])
app.include_router(prediction_router, prefix="/predict", tags=["Prediction"])


@app.get("/", tags=["Root"])
def root():
    return {
        "project": "HR Salary Prediction",
        "algorithm": "Simple Linear Regression",
        "docs": "/docs",
    }
