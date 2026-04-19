"""Pydantic response schema for salary prediction."""

from pydantic import BaseModel


class SalaryPredictionResponse(BaseModel):
    years_experience: float
    predicted_salary: float
    currency: str = "USD"

    model_config = {
        "json_schema_extra": {
            "example": {
                "years_experience": 5.0,
                "predicted_salary": 72000.50,
                "currency": "USD",
            }
        }
    }
