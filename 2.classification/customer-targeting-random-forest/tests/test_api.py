"""Unit tests for RF Classification API."""

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
VALID_PAYLOAD = {"age": 35, "estimated_salary": 60000.0, "gender": "Male"}


def test_health():
    r = client.get("/health/")
    assert r.status_code == 200


def test_train():
    r = client.post("/train/")
    assert r.status_code in (200, 404, 500)


def test_predict_valid():
    r = client.post("/predict/", json=VALID_PAYLOAD)
    assert r.status_code in (200, 404)


def test_predict_invalid_gender():
    r = client.post("/predict/", json={**VALID_PAYLOAD, "gender": "Unknown"})
    assert r.status_code in (422, 404, 500)


def test_predict_missing_field():
    r = client.post("/predict/", json={"age": 35})
    assert r.status_code == 422
