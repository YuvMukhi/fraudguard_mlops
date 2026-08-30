"""
Unit tests for Phase 9: CI/CD Automated Eval-Gated Model Promotion Pipeline.
"""
import os
import json
import pytest
import numpy as np
import pandas as pd

from src.models.eval_promotion import ModelEvaluatorPromoter, evaluate_and_promote


class MockCandidateModel:
    """High-performing model fixture."""
    def __init__(self, best_threshold: float = 0.5):
        self.best_threshold = best_threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        v1 = X["V1"].values if "V1" in X.columns else np.zeros(len(X))
        probs = np.where(v1 > 1.0, 0.95, 0.05)
        return np.column_stack([1.0 - probs, probs])


class MockChampionModel:
    """Weak / uninformative champion model fixture."""
    def __init__(self, best_threshold: float = 0.5):
        self.best_threshold = best_threshold

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.full(len(X), 0.50)
        return np.column_stack([1.0 - probs, probs])


@pytest.fixture
def temp_eval_environment(tmp_path):
    """Set up temporary dataset, models directory, and isolated tracking DB."""
    data_dir = tmp_path / "data" / "processed"
    model_dir = tmp_path / "models"
    data_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    tracking_uri = f"sqlite:///{tmp_path}/test_mlflow.db"

    # Create dummy test dataset
    np.random.seed(42)
    n_samples = 100
    y = np.array([0] * 80 + [1] * 20)
    v1 = np.random.normal(0, 0.5, n_samples)
    v1[y == 1] += 3.0  # Positive class has V1 > 1.0

    df = pd.DataFrame({
        "Time": np.linspace(0, 1000, n_samples),
        "V1": v1,
        "V2": np.random.normal(0, 1, n_samples),
        "Amount": np.random.uniform(1, 100, n_samples),
        "Class": y
    })

    df.to_parquet(data_dir / "test.parquet")
    df.to_parquet(data_dir / "val.parquet")
    df.to_parquet(data_dir / "train.parquet")

    return str(data_dir), str(model_dir), tracking_uri


def test_evaluate_model_on_data(temp_eval_environment):
    data_dir, model_dir, tracking_uri = temp_eval_environment
    promoter = ModelEvaluatorPromoter(data_dir=data_dir, model_dir=model_dir, tracking_uri=tracking_uri)

    model = MockCandidateModel()
    X = pd.DataFrame({"V1": np.array([-2.0, 2.0]), "Amount": [10.0, 20.0]})
    y = pd.Series([0, 1])

    metrics = promoter.evaluate_model_on_data(model, X, y)
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "f1_score" in metrics
    assert metrics["roc_auc"] >= 0.0 and metrics["roc_auc"] <= 1.0


def test_eval_promotion_gated_success(temp_eval_environment):
    data_dir, model_dir, tracking_uri = temp_eval_environment
    promoter = ModelEvaluatorPromoter(data_dir=data_dir, model_dir=model_dir, tracking_uri=tracking_uri, auc_threshold_delta=0.01)

    candidate = MockCandidateModel()
    champion = MockChampionModel()

    report = promoter.run_evaluation_and_promotion(
        candidate_model=candidate,
        champion_model=champion,
        candidate_name="test_candidate_v2"
    )

    assert report["promoted"] is True
    assert report["decision"] == "PROMOTED_TO_CHAMPION"
    assert report["roc_auc_delta"] >= 0.01
    assert os.path.exists(os.path.join(model_dir, "promotion_report.json"))


def test_eval_promotion_gated_rejection(temp_eval_environment):
    data_dir, model_dir, tracking_uri = temp_eval_environment
    promoter = ModelEvaluatorPromoter(data_dir=data_dir, model_dir=model_dir, tracking_uri=tracking_uri, auc_threshold_delta=0.05)

    # Candidate and champion have identical metrics
    candidate = MockCandidateModel()
    champion = MockCandidateModel()

    report = promoter.run_evaluation_and_promotion(
        candidate_model=candidate,
        champion_model=champion,
        candidate_name="test_identical_candidate"
    )

    assert report["promoted"] is False
    assert report["decision"] == "REJECTED_UNDERPERFORMING"
    assert report["roc_auc_delta"] < 0.05
