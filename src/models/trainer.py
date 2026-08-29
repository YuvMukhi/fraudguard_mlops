"""
MLflow Experiment Tracking and Model Registry Training Script.
Trains Baseline and Heavy models, logs metrics, artifacts, and registers models in MLflow.
"""
import os
import time
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from src.models.baseline import BaselineModel
from src.models.heavy import HeavyModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_processed_data(data_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, val, test splits from parquet/csv files."""
    train_path = os.path.join(data_dir, "train.parquet")
    val_path = os.path.join(data_dir, "val.parquet")
    test_path = os.path.join(data_dir, "test.parquet")

    if not os.path.exists(train_path):
        train_path = os.path.join(data_dir, "train.csv")
        val_path = os.path.join(data_dir, "val.csv")
        test_path = os.path.join(data_dir, "test.csv")

    train_df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
    val_df = pd.read_parquet(val_path) if val_path.endswith(".parquet") else pd.read_csv(val_path)
    test_df = pd.read_parquet(test_path) if test_path.endswith(".parquet") else pd.read_csv(test_path)

    return train_df, val_df, test_df


def train_and_track_experiments(
    data_dir: str = "data/processed",
    output_model_dir: str = "models",
    experiment_name: str = "fraud-detection-experiment",
    registered_model_name: str = "fraud-detection-model"
) -> Dict[str, Any]:
    """Execute training pipeline with MLflow tracking and registry."""
    os.makedirs(output_model_dir, exist_ok=True)
    
    # Configure MLflow tracking
    tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info(f"Loading data from {data_dir}...")
    train_df, val_df, test_df = load_processed_data(data_dir)

    feature_cols = [c for c in train_df.columns if c != "Class"]
    target_col = "Class"

    X_train, y_train = train_df[feature_cols], train_df[target_col]
    X_val, y_val = val_df[feature_cols], val_df[target_col]
    X_test, y_test = test_df[feature_cols], test_df[target_col]

    results = {}
    sample_x = X_test.iloc[0:1]

    # ================================
    # 1. Train Baseline Model
    # ================================
    logger.info("--- Starting Baseline Model Experiment ---")
    with mlflow.start_run(run_name="baseline_logistic_regression") as run:
        baseline = BaselineModel(C=1.0, class_weight="balanced", random_state=42)
        
        start_time = time.time()
        baseline.fit(X_train, y_train, X_val, y_val)
        train_time = time.time() - start_time

        # Measure single-sample inference latency
        latencies = []
        for _ in range(100):
            t0 = time.time()
            _ = baseline.predict_proba(sample_x)
            latencies.append((time.time() - t0) * 1000)
        avg_latency_ms = float(np.mean(latencies))

        # Evaluate metrics
        val_metrics = baseline.evaluate(X_val, y_val)
        test_metrics = baseline.evaluate(X_test, y_test)

        # Log MLflow parameters & metrics
        mlflow.log_param("model_type", "baseline_logistic_regression")
        mlflow.log_param("C", baseline.C)
        mlflow.log_param("class_weight", baseline.class_weight)
        mlflow.log_param("best_threshold", baseline.best_threshold)

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("avg_latency_ms", avg_latency_ms)

        # Save local checkpoint & log artifact
        local_path = os.path.join(output_model_dir, "baseline_model.joblib")
        baseline.save(local_path)
        mlflow.log_artifact(local_path, artifact_path="model_files")

        run_id = run.info.run_id
        results["baseline"] = {
            "run_id": run_id,
            "metrics": test_metrics,
            "latency_ms": avg_latency_ms,
            "checkpoint_path": local_path
        }

    # ================================
    # 2. Train Heavy Model
    # ================================
    logger.info("--- Starting Heavy Model Experiment ---")
    with mlflow.start_run(run_name="heavy_pytorch_dnn") as run:
        heavy = HeavyModel(input_dim=len(feature_cols), epochs=10, batch_size=256, pos_weight=50.0, random_state=42)

        start_time = time.time()
        heavy.fit(X_train, y_train, X_val, y_val)
        train_time = time.time() - start_time

        # Measure latency
        latencies = []
        for _ in range(100):
            t0 = time.time()
            _ = heavy.predict_proba(sample_x)
            latencies.append((time.time() - t0) * 1000)
        avg_latency_ms = float(np.mean(latencies))

        # Evaluate metrics
        val_metrics = heavy.evaluate(X_val, y_val)
        test_metrics = heavy.evaluate(X_test, y_test)

        # Log MLflow parameters & metrics
        mlflow.log_param("model_type", "heavy_pytorch_dnn")
        mlflow.log_param("epochs", heavy.epochs)
        mlflow.log_param("batch_size", heavy.batch_size)
        mlflow.log_param("pos_weight", heavy.pos_weight)
        mlflow.log_param("best_threshold", heavy.best_threshold)

        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("avg_latency_ms", avg_latency_ms)

        # Save local checkpoint & log artifact
        local_path = os.path.join(output_model_dir, "heavy_model.pt")
        heavy.save(local_path)
        mlflow.log_artifact(local_path, artifact_path="model_files")

        run_id = run.info.run_id
        results["heavy"] = {
            "run_id": run_id,
            "metrics": test_metrics,
            "latency_ms": avg_latency_ms,
            "checkpoint_path": local_path
        }

    # Save summary manifest
    with open(os.path.join(output_model_dir, "training_manifest.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Training and experiment tracking completed successfully!")
    return results


if __name__ == "__main__":
    train_and_track_experiments()
