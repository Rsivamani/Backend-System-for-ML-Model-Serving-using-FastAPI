"""Pydantic request/response schemas for Profit Prediction (MLR/SVM-R)."""

from pydantic import BaseModel, Field


class ProfitPredictionRequest(BaseModel):
    rd_spend: float = Field(..., ge=0.0, example=165349.20, description="R&D Spend in USD")
    administration: float = Field(..., ge=0.0, example=136897.80, description="Administration costs in USD")
    marketing_spend: float = Field(..., ge=0.0, example=471784.10, description="Marketing Spend in USD")
    state: str = Field(..., example="New York", description="State: 'New York', 'California', or 'Florida'")

    model_config = {
        "json_schema_extra": {
            "example": {
                "rd_spend": 165349.20,
                "administration": 136897.80,
                "marketing_spend": 471784.10,
                "state": "New York",
            }
        }
    }


class ProfitPredictionResponse(BaseModel):
    rd_spend: float
    administration: float
    marketing_spend: float
    state: str
    predicted_profit: float
    currency: str = "USD"

    model_config = {
        "json_schema_extra": {
            "example": {
                "rd_spend": 165349.20,
                "administration": 136897.80,
                "marketing_spend": 471784.10,
                "state": "New York",
                "predicted_profit": 192261.83,
                "currency": "USD",
            }
        }
    }
