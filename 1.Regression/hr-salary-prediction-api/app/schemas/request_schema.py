"""Pydantic request schema for salary prediction."""

from pydantic import BaseModel, Field, field_validator


class SalaryPredictionRequest(BaseModel):
    years_experience: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        example=5.0,
        description="Years of professional experience (0–50).",
    )

    @field_validator("years_experience")
    @classmethod
    def must_be_positive(cls, v: float) -> float:
        if v < 0:
            raise ValueError("years_experience must be ≥ 0")
        return v

    model_config = {"json_schema_extra": {"example": {"years_experience": 5.0}}}
