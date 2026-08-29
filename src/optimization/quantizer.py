"""
Model Optimization & Quantization Module.
Converts PyTorch Heavy Model to ONNX and INT8 Dynamic Quantization.
Includes latency benchmarking (P50, P95, P99), size reduction analysis, and accuracy fallback validation.
"""
import os
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

try:
    import torch.onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

from src.models.heavy import HeavyModel, PyTorchFraudNN

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelOptimizer:
    """Manages PyTorch DNN quantization, ONNX export, and runtime performance benchmarking."""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.fp32_model_path = os.path.join(self.model_dir, "heavy_model.pt")
        self.onnx_fp32_path = os.path.join(self.model_dir, "heavy_model.onnx")
        self.onnx_int8_path = os.path.join(self.model_dir, "heavy_model_int8.onnx")
        self.quantized_pt_path = os.path.join(self.model_dir, "heavy_model_quantized.pt")

    def quantize_pytorch_model(self, heavy_model: HeavyModel) -> str:
        """Apply PyTorch dynamic INT8 quantization to linear layers."""
        logger.info("Applying PyTorch Dynamic INT8 Quantization...")
        nn_model = heavy_model.model.cpu()
        nn_model.eval()

        quantized_nn = torch.quantization.quantize_dynamic(
            nn_model,
            {nn.Linear},
            dtype=torch.qint8
        )

        torch.save(quantized_nn.state_dict(), self.quantized_pt_path)
        logger.info(f"Saved PyTorch INT8 quantized model to {self.quantized_pt_path}")
        return self.quantized_pt_path

    def export_onnx(self, heavy_model: HeavyModel) -> str:
        """Export PyTorch DNN model to ONNX FP32 format."""
        logger.info("Exporting PyTorch model to ONNX format...")
        input_dim = heavy_model.input_dim
        dummy_input = torch.randn(1, input_dim, dtype=torch.float32)
        
        nn_model = heavy_model.model.cpu()
        nn_model.eval()

        torch.onnx.export(
            nn_model,
            dummy_input,
            self.onnx_fp32_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
        )
        logger.info(f"Successfully exported ONNX FP32 model to {self.onnx_fp32_path}")
        return self.onnx_fp32_path

    def benchmark_models(
        self,
        heavy_model: HeavyModel,
        test_df: pd.DataFrame,
        n_iterations: int = 100,
        max_error_tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Benchmark latency (P50, P95, P99), model file sizes, and accuracy agreement.
        Includes automatic fallback validation.
        """
        feature_cols = [c for c in test_df.columns if c != "Class"]
        X_sample = test_df[feature_cols].values.astype(np.float32)
        
        # 1. Unoptimized PyTorch FP32 Benchmark
        pt_latencies = []
        for _ in range(n_iterations):
            idx = np.random.randint(0, len(X_sample))
            sample = X_sample[idx:idx+1]
            t0 = time.time()
            _ = heavy_model.predict_proba(pd.DataFrame(sample, columns=feature_cols))
            pt_latencies.append((time.time() - t0) * 1000.0)

        pt_probs = heavy_model.predict_proba(pd.DataFrame(X_sample, columns=feature_cols))

        # 2. PyTorch INT8 Quantized Model Benchmark
        quantized_pt_path = self.quantize_pytorch_model(heavy_model)
        quant_latencies = []
        
        # Load PyTorch quantized model
        quantized_nn = torch.quantization.quantize_dynamic(
            heavy_model.model.cpu(),
            {nn.Linear},
            dtype=torch.qint8
        )
        quantized_nn.eval()

        for _ in range(n_iterations):
            idx = np.random.randint(0, len(X_sample))
            sample = torch.tensor(X_sample[idx:idx+1], dtype=torch.float32)
            t0 = time.time()
            with torch.no_grad():
                _ = torch.sigmoid(quantized_nn(sample)).numpy()
            quant_latencies.append((time.time() - t0) * 1000.0)

        with torch.no_grad():
            quant_probs = torch.sigmoid(quantized_nn(torch.tensor(X_sample, dtype=torch.float32))).numpy().flatten()

        # Compute accuracy agreement between FP32 and Quantized INT8
        abs_diffs = np.abs(pt_probs - quant_probs)
        mean_abs_diff = float(np.mean(abs_diffs))
        max_abs_diff = float(np.max(abs_diffs))

        # Fallback decision
        fallback_triggered = mean_abs_diff > max_error_tolerance
        active_model_type = "PyTorch FP32 (Fallback)" if fallback_triggered else "PyTorch INT8 Quantized"

        if fallback_triggered:
            logger.warning(
                f"Quantization error ({mean_abs_diff:.4f}) exceeded tolerance ({max_error_tolerance}). "
                "Triggering fallback to FP32 model."
            )
        else:
            logger.info(f"Quantized model validated successfully! Mean absolute difference: {mean_abs_diff:.4f}")

        # Compute latency stats
        def get_stats(latencies):
            return {
                "p50_ms": round(float(np.percentile(latencies, 50)), 3),
                "p95_ms": round(float(np.percentile(latencies, 95)), 3),
                "p99_ms": round(float(np.percentile(latencies, 99)), 3),
                "mean_ms": round(float(np.mean(latencies)), 3)
            }

        fp32_size_kb = os.path.getsize(self.fp32_model_path) / 1024.0 if os.path.exists(self.fp32_model_path) else 0.0
        int8_size_kb = os.path.getsize(quantized_pt_path) / 1024.0 if os.path.exists(quantized_pt_path) else 0.0
        size_reduction_pct = round((1.0 - (int8_size_kb / fp32_size_kb)) * 100.0, 2) if fp32_size_kb > 0 else 0.0

        return {
            "unoptimized_fp32": {
                "file_size_kb": round(fp32_size_kb, 2),
                "latency": get_stats(pt_latencies)
            },
            "quantized_int8": {
                "file_size_kb": round(int8_size_kb, 2),
                "latency": get_stats(quant_latencies),
                "size_reduction_pct": size_reduction_pct
            },
            "validation": {
                "mean_abs_probability_diff": round(mean_abs_diff, 4),
                "max_abs_probability_diff": round(max_abs_diff, 4),
                "error_tolerance_spec": max_error_tolerance,
                "fallback_triggered": fallback_triggered,
                "active_serving_model": active_model_type
            }
        }
