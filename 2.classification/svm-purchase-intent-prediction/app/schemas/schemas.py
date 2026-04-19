"""Shared Pydantic schemas for SVM Classification."""

from pydantic import BaseModel, Field


class PurchasePredictionRequest(BaseModel):
    age: int = Field(..., ge=18, le=100, example=35)
    estimated_salary: float = Field(..., ge=0.0, example=60000.0)
    gender: str = Field(..., example="Male", description="'Male' or 'Female'")


class PurchasePredictionResponse(BaseModel):
    age: int
    estimated_salary: float
    gender: str
    purchased: int
    purchase_probability: float
    confidence: float
    prediction_label: str
