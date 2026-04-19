"""Schemas for DT Regression Profit Prediction."""

from pydantic import BaseModel, Field


class ProfitPredictionRequest(BaseModel):
    rd_spend: float = Field(..., ge=0.0, example=165349.20)
    administration: float = Field(..., ge=0.0, example=136897.80)
    marketing_spend: float = Field(..., ge=0.0, example=471784.10)
    state: str = Field(..., example="New York", description="New York, California, or Florida")


class ProfitPredictionResponse(BaseModel):
    rd_spend: float
    administration: float
    marketing_spend: float
    state: str
    predicted_profit: float
    currency: str = "USD"
