"""Health-check route."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/", summary="Health Check")
def health_check():
    """Returns API liveness status."""
    return {"status": "ok", "service": "HR Salary Prediction API"}
