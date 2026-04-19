"""Pydantic schemas for Social Network Ads Classification."""

from pydantic import BaseModel, Field
from typing import Optional


class PurchasePredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35, description="User age (18–100)")
    estimated_salary: float = Field(..., ge=0.0, example=60000.0, description="Annual estimated salary in USD")
    gender: str = Field(..., example="Male", description="Gender: 'Male' or 'Female'")

    model_config = {
        "json_schema_extra": {
            "example": {"age": 35, "estimated_salary": 60000.0, "gender": "Male"}
        }
    }


class PurchasePredictionResponse(BaseModel):
    age: int
    estimated_salary: float
    gender: str
    purchased: int
    purchase_probability: float
    confidence: float
    prediction_label: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35, "estimated_salary": 60000.0, "gender": "Male",
                "purchased": 1, "purchase_probability": 0.87,
                "confidence": 0.87, "prediction_label": "Will Purchase",
            }
        }
    }
