"""
Unit tests for Statistical Drift Detection (KS Test, PSI) and Prometheus Observability.
"""
import pytest
import numpy as np
from fastapi.testclient import TestClient

from src.monitoring.drift_detector import DriftDetector, calculate_psi
from src.serving.api import app


def test_psi_identical_distribution():
    # Identical distributions should yield near zero PSI (< 0.05)
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    prod = np.random.normal(0, 1, 1000)

    psi = calculate_psi(ref, prod, num_bins=10)
    assert psi < 0.05


def test_psi_shifted_distribution():
    # Shifted mean distribution should yield high PSI (> 0.25)
    np.random.seed(42)
    ref = np.random.normal(0, 1, 1000)
    prod = np.random.normal(2.5, 1, 1000)

    psi = calculate_psi(ref, prod, num_bins=10)
    assert psi >= 0.25


def test_drift_detector_buffer_and_report():
    detector = DriftDetector(data_dir="data/processed")
    
    # Add samples without drift
    for _ in range(50):
        sample = {f"V{i}": np.random.normal(0, 1) for i in range(1, 29)}
        sample["Time"] = 100.0
        sample["Amount"] = 50.0
        detector.add_production_sample(sample)

    report = detector.compute_drift_report()
    assert report["status"] in ["no_drift", "moderate_drift", "significant_drift"]
    assert report["sample_count"] == 50
    assert "features" in report


def test_prometheus_metrics_endpoint():
    with TestClient(app) as client:
        response = client.get("/metrics")
        assert response.status_code == 200
        content = response.text
        assert "http_requests_total" in content or "http_request_duration_seconds" in content


def test_monitoring_drift_endpoint():
    with TestClient(app) as client:
        # Trigger single prediction to populate drift buffer
        sample_txn = {
            "Time": 10.0, "V1": 0.1, "V2": 0.2, "V3": 0.3, "V4": 0.4, "V5": 0.5,
            "V6": 0.1, "V7": 0.2, "V8": 0.3, "V9": 0.4, "V10": 0.5,
            "V11": 0.1, "V12": 0.2, "V13": 0.3, "V14": 0.4, "V15": 0.5,
            "V16": 0.1, "V17": 0.2, "V18": 0.3, "V19": 0.4, "V20": 0.5,
            "V21": 0.1, "V22": 0.2, "V23": 0.3, "V24": 0.4, "V25": 0.5,
            "V26": 0.1, "V27": 0.2, "V28": 0.3, "Amount": 50.0
        }
        client.post("/predict", json={"transaction": sample_txn})

        response = client.get("/monitoring/drift")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "max_psi" in data
