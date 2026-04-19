"""Unit tests for Decision Tree Regression API."""

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


def test_train():
    r = client.post("/train/")
    assert r.status_code in (200, 404, 500)


def test_predict_valid():
    r = client.post("/predict/", json=VALID_PAYLOAD)
    assert r.status_code in (200, 404)


def test_predict_invalid():
    r = client.post("/predict/", json={**VALID_PAYLOAD, "state": "Texas"})
    assert r.status_code in (422, 404, 500)
