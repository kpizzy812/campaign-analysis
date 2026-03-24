import pytest
from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "model_version" in data
    assert "trained_at" in data


def test_predict_single(client):
    payload = {
        "geo": "US", "vertical": "nutra", "traffic_source": "facebook",
        "device": "mobile", "os": "android",
        "bid": 0.5, "daily_budget": 100,
        "hour": 14, "dow": 2,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "probability" in data
    assert "prediction" in data
    assert 0 <= data["probability"] <= 1


def test_predict_batch(client):
    items = [
        {"geo": "US", "vertical": "nutra", "traffic_source": "facebook",
         "device": "mobile", "os": "android",
         "bid": 0.5, "daily_budget": 100, "hour": 14, "dow": 2},
        {"geo": "DE", "vertical": "gambling", "traffic_source": "google",
         "device": "desktop", "os": "windows",
         "bid": 1.0, "daily_budget": 200, "hour": 10, "dow": 5},
    ]
    r = client.post("/predict/batch", json=items)
    assert r.status_code == 200
    data = r.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2


def test_predict_invalid_data(client):
    payload = {"geo": "US"}  # missing fields
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_batch_too_large(client):
    items = [{"geo": "US", "vertical": "nutra", "traffic_source": "facebook",
              "device": "mobile", "os": "android",
              "bid": 0.5, "daily_budget": 100, "hour": 14, "dow": 2}] * 10_001
    r = client.post("/predict/batch", json=items)
    assert r.status_code == 422
