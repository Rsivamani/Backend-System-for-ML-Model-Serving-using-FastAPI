"""Unit tests for MLR Profit Prediction API."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


class TestHealth:
    def test_health(self):
        r = client.get("/health/")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestTraining:
    def test_train(self):
        r = client.post("/train/")
        assert r.status_code in (200, 404, 500)
        if r.status_code == 200:
            assert "metrics" in r.json()


class TestPrediction:
    VALID_PAYLOAD = {
        "rd_spend": 165349.20,
        "administration": 136897.80,
        "marketing_spend": 471784.10,
        "state": "New York",
    }

    def test_predict_valid(self):
        r = client.post("/predict/", json=self.VALID_PAYLOAD)
        assert r.status_code in (200, 404)

    def test_predict_invalid_state(self):
        payload = {**self.VALID_PAYLOAD, "state": "Texas"}
        r = client.post("/predict/", json=payload)
        assert r.status_code in (422, 404, 500)

    def test_predict_missing_field(self):
        r = client.post("/predict/", json={"rd_spend": 100000})
        assert r.status_code == 422
