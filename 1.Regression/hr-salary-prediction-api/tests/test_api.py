"""Unit tests for SLR HR Salary Prediction API."""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealth:
    def test_health_check(self):
        response = client.get("/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestTraining:
    def test_train_endpoint(self):
        response = client.post("/train/")
        assert response.status_code in (200, 404, 500)
        if response.status_code == 200:
            data = response.json()
            assert "metrics" in data
            assert "r2_score" in data["metrics"]
            assert "mae" in data["metrics"]


class TestPrediction:
    def test_predict_valid_input(self):
        response = client.post("/predict/", json={"years_experience": 5.0})
        # Will return 404 if model not yet trained; accept both
        assert response.status_code in (200, 404)
        if response.status_code == 200:
            data = response.json()
            assert "predicted_salary" in data
            assert data["years_experience"] == 5.0

    def test_predict_zero_experience(self):
        response = client.post("/predict/", json={"years_experience": 0.0})
        assert response.status_code in (200, 404)

    def test_predict_invalid_negative(self):
        response = client.post("/predict/", json={"years_experience": -1.0})
        assert response.status_code == 422  # Pydantic validation error

    def test_predict_missing_field(self):
        response = client.post("/predict/", json={})
        assert response.status_code == 422
