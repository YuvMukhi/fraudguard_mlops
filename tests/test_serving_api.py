"""
Unit and Integration Tests for FastAPI Serving Layer.
Tests /health, /predict, /predict/batch, /models, and version pinning functionality.
"""
import pytest
from fastapi.testclient import TestClient
from src.serving.api import app, TransactionItem


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_transaction():
    return {
        "Time": 100.0,
        "V1": -1.2, "V2": 0.5, "V3": 0.8, "V4": 2.1, "V5": -0.4,
        "V6": -0.1, "V7": -0.9, "V8": 0.3, "V9": -0.5, "V10": 2.8,
        "V11": 1.9, "V12": -1.2, "V13": 0.0, "V14": -3.0, "V15": 0.1,
        "V16": 0.7, "V17": -1.8, "V18": -0.2, "V19": 0.4, "V20": 0.1,
        "V21": 0.2, "V22": -0.1, "V23": 0.04, "V24": 0.25, "V25": -0.15,
        "V26": 0.1, "V27": 0.05, "V28": -0.01,
        "Amount": 99.99
    }


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["uptime_seconds"] >= 0
    assert isinstance(data["loaded_models"], list)


def test_predict_single_endpoint_champion(client, sample_transaction):
    payload = {
        "transaction": sample_transaction,
        "model_version": "champion"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0]["is_fraud"] in [0, 1]
    assert 0.0 <= data["predictions"][0]["fraud_probability"] <= 1.0
    assert "champion" in data["model_version"].lower() or "heavy" in data["model_version"].lower()
    assert data["latency_ms"] > 0.0


def test_predict_single_endpoint_baseline_pinning(client, sample_transaction):
    payload = {
        "transaction": sample_transaction,
        "model_version": "baseline"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert "baseline" in data["model_version"].lower() or "logistic" in data["model_version"].lower()


def test_predict_batch_endpoint(client, sample_transaction):
    payload = {
        "transactions": [sample_transaction, sample_transaction],
        "model_version": "champion"
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert pred["is_fraud"] in [0, 1]
        assert 0.0 <= pred["fraud_probability"] <= 1.0


def test_list_models_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "active_models" in data
    assert "champion" in data["active_models"] or "baseline" in data["active_models"]
