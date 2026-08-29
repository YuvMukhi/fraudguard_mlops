"""
Model Registry Loader & Inference Engine for FastAPI Serving Layer.
Optimized for ultra-low latency inference using NumPy arrays for single and batch predictions.
Supports MLflow Model Registry version pinning and local fallback checkpoints.
"""
import os
import json
import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import pandas as pd
import joblib
import mlflow

from src.models.baseline import BaselineModel
from src.models.heavy import HeavyModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelRegistryLoader:
    """Thread-safe model registry loader supporting version pinning and local fallback."""

    def __init__(self, model_dir: str = "models", data_dir: str = "data/processed"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.loaded_models: Dict[str, Any] = {}
        self.scaler = None
        self.feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        self._load_scaler()

    def _load_scaler(self):
        """Load feature scaler artifact."""
        scaler_path = os.path.join(self.data_dir, "scaler.joblib")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded feature scaler from {scaler_path}")
        else:
            logger.warning(f"Scaler artifact not found at {scaler_path}. Features must be pre-scaled.")

    def get_model(self, version_or_alias: str = "champion") -> Tuple[Any, str, float]:
        """
        Load model instance from memory cache, MLflow registry, or local file checkpoint.
        Returns: (model_instance, resolved_version_string, best_threshold)
        """
        key = str(version_or_alias).lower()

        # Check in-memory cache
        if key in self.loaded_models:
            return self.loaded_models[key]

        model_instance = None
        resolved_version = key
        threshold = 0.5

        # 1. Attempt loading from MLflow Registry
        try:
            tracking_uri = "sqlite:///mlflow.db"
            mlflow.set_tracking_uri(tracking_uri)
            
            if key in ["champion", "heavy", "production", "default"]:
                model_uri = f"models:/fraud-detection-model@champion"
                resolved_version = "champion (Heavy DNN)"
            elif key in ["baseline", "challenger"]:
                model_uri = f"models:/fraud-detection-model@baseline"
                resolved_version = "baseline (Logistic Regression)"
            elif key.isdigit():
                model_uri = f"models:/fraud-detection-model/{key}"
                resolved_version = f"version_{key}"
            else:
                model_uri = f"models:/fraud-detection-model/{key}"

            logger.info(f"Attempting to load model from MLflow registry URI: {model_uri}")
            model_instance = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Successfully loaded model from MLflow registry: {resolved_version}")
        except Exception as e:
            logger.warning(f"Could not load from MLflow registry ({e}). Falling back to local disk checkpoints.")

        # 2. Local fallback if MLflow load failed
        if model_instance is None:
            if key in ["baseline", "challenger", "1"]:
                local_path = os.path.join(self.model_dir, "baseline_model.joblib")
                if os.path.exists(local_path):
                    model_instance = BaselineModel.load(local_path)
                    resolved_version = "baseline_local_v1"
                    threshold = model_instance.best_threshold
            else:
                local_path = os.path.join(self.model_dir, "heavy_model.pt")
                if os.path.exists(local_path):
                    model_instance = HeavyModel.load(local_path)
                    resolved_version = "heavy_local_v2"
                    threshold = model_instance.best_threshold

        if model_instance is None:
            logger.warning("No saved model checkpoints found. Initializing inline baseline model.")
            model_instance = BaselineModel()
            resolved_version = "inline_fallback_baseline"

        result = (model_instance, resolved_version, threshold)
        self.loaded_models[key] = result
        return result

    def predict(
        self,
        features: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame, np.ndarray],
        version_or_alias: str = "champion",
        custom_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        High-Performance Prediction Pipeline:
        Converts feature dictionaries directly to contiguous float32 NumPy arrays to bypass pandas overhead.
        """
        t0 = time.time()
        model, model_ver, default_thresh = self.get_model(version_or_alias)
        threshold = custom_threshold if custom_threshold is not None else default_thresh

        # Zero-overhead NumPy matrix construction
        if isinstance(features, dict):
            matrix = np.array([[features.get(c, 0.0) for c in self.feature_cols]], dtype=np.float32)
        elif isinstance(features, list):
            matrix = np.array([[f.get(c, 0.0) for c in self.feature_cols] for f in features], dtype=np.float32)
        elif isinstance(features, pd.DataFrame):
            matrix = features[self.feature_cols].values.astype(np.float32)
        else:
            matrix = np.asarray(features, dtype=np.float32)

        # Get probabilities
        if hasattr(model, "predict_proba"):
            # Pass DataFrame if object is sklearn or custom wrapper requiring DataFrame
            if isinstance(model, (BaselineModel, HeavyModel)):
                probs = model.predict_proba(pd.DataFrame(matrix, columns=self.feature_cols))
            else:
                try:
                    probs = model.predict_proba(matrix)
                except Exception:
                    probs = model.predict_proba(pd.DataFrame(matrix, columns=self.feature_cols))

            if isinstance(probs, np.ndarray) and probs.ndim == 2:
                probs = probs[:, 1]
        elif hasattr(model, "predict"):
            raw_preds = model.predict(matrix)
            probs = np.array(raw_preds, dtype=float)
        else:
            raise ValueError("Loaded model object does not support predict_proba or predict interface.")

        probs = np.clip(probs, 0.0, 1.0)
        predictions = (probs >= threshold).astype(int).tolist()
        probabilities = [float(p) for p in probs]

        latency_ms = (time.time() - t0) * 1000.0

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "model_version": model_ver,
            "threshold_used": float(threshold),
            "latency_ms": float(latency_ms)
        }
