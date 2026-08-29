"""
Unit tests for Model Quantization, ONNX Export, Benchmarking, and Accuracy Fallback.
"""
import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.models.heavy import HeavyModel
from src.optimization.quantizer import ModelOptimizer
from src.serving.api import app


@pytest.fixture
def sample_heavy_model():
    model = HeavyModel(input_dim=30, epochs=2, batch_size=32)
    X = pd.DataFrame(np.random.randn(100, 30))
    y = pd.Series([0]*90 + [1]*10)
    model.fit(X, y, X, y)
    return model


def test_pytorch_dynamic_quantization(sample_heavy_model, tmp_path):
    optimizer = ModelOptimizer(model_dir=str(tmp_path))
    quant_path = optimizer.quantize_pytorch_model(sample_heavy_model)

    assert os.path.exists(quant_path)
    assert os.path.getsize(quant_path) > 0


def test_onnx_export(sample_heavy_model, tmp_path):
    optimizer = ModelOptimizer(model_dir=str(tmp_path))
    onnx_path = optimizer.export_onnx(sample_heavy_model)

    assert os.path.exists(onnx_path)
    assert os.path.getsize(onnx_path) > 0


def test_model_benchmark_and_fallback(sample_heavy_model, tmp_path):
    optimizer = ModelOptimizer(model_dir=str(tmp_path))
    # Save base model file so file size calculation works
    sample_heavy_model.save(optimizer.fp32_model_path)

    test_df = pd.DataFrame(
        np.random.randn(50, 31),
        columns=["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    )

    results = optimizer.benchmark_models(heavy_model=sample_heavy_model, test_df=test_df, n_iterations=10)

    assert "unoptimized_fp32" in results
    assert "quantized_int8" in results
    assert "validation" in results
    assert "p50_ms" in results["unoptimized_fp32"]["latency"]
    assert "p50_ms" in results["quantized_int8"]["latency"]
    assert results["validation"]["fallback_triggered"] in [True, False]


def test_optimization_benchmark_endpoint():
    with TestClient(app) as client:
        response = client.get("/optimization/benchmark")
        assert response.status_code == 200
        data = response.json()
        assert "unoptimized_fp32" in data
        assert "quantized_int8" in data
        assert "validation" in data
