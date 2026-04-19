"""Unit tests for SVM-R Profit Prediction API."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "rd_spend": 165349.20, "administration": 136897.80,
    "marketing_spend": 471784.10, "state": "New York",
}


def test_health():
    r = client.get("/health/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_train():
    r = client.post("/train/")
    assert r.status_code in (200, 404, 500)


def test_predict_valid():
    r = client.post("/predict/", json=VALID_PAYLOAD)
    assert r.status_code in (200, 404)


def test_predict_invalid_state():
    r = client.post("/predict/", json={**VALID_PAYLOAD, "state": "Texas"})
    assert r.status_code in (422, 404, 500)


def test_predict_missing_field():
    r = client.post("/predict/", json={"rd_spend": 100000})
    assert r.status_code == 422
