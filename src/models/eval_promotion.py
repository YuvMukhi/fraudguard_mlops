"""
Automated Model Re-Evaluation and Promotion Pipeline.
Evaluates candidate model performance against the current champion model.
Promotes candidate to Champion in MLflow Model Registry ONLY if ROC-AUC improves by > threshold (default 0.01).
"""
import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Tuple, Optional

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, average_precision_score
import mlflow
from mlflow.tracking import MlflowClient

from src.models.baseline import BaselineModel
from src.models.heavy import HeavyModel
from src.models.trainer import load_processed_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_evaluation_data(data_dir: str = "data/processed") -> pd.DataFrame:
    """Load evaluation test split from parquet or csv."""
    test_path = os.path.join(data_dir, "test.parquet")
    if not os.path.exists(test_path):
        test_path = os.path.join(data_dir, "test.csv")
    if not os.path.exists(test_path):
        test_path = os.path.join(data_dir, "val.parquet")
    if not os.path.exists(test_path):
        test_path = os.path.join(data_dir, "val.csv")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"No evaluation dataset found in {data_dir} (expected test.parquet or test.csv).")

    return pd.read_parquet(test_path) if test_path.endswith(".parquet") else pd.read_csv(test_path)


class ModelEvaluatorPromoter:
    """Evaluates candidate models against current champion and handles gated promotion."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        model_dir: str = "models",
        tracking_uri: str = "sqlite:///mlflow.db",
        model_name: str = "fraud-detection-model",
        auc_threshold_delta: float = 0.01
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.tracking_uri = tracking_uri
        self.model_name = model_name
        self.auc_threshold_delta = auc_threshold_delta

        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(tracking_uri=self.tracking_uri)

    def evaluate_model_on_data(self, model: Any, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics for a model instance."""
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            if isinstance(probs, np.ndarray) and probs.ndim == 2:
                probs = probs[:, 1]
        elif hasattr(model, "predict"):
            probs = model.predict(X)
        else:
            raise ValueError("Model does not support predict_proba or predict interface.")

        probs = np.clip(probs, 0.0, 1.0)
        threshold = getattr(model, "best_threshold", 0.5)
        preds = (probs >= threshold).astype(int)

        roc_auc = float(roc_auc_score(y, probs))
        pr_auc = float(average_precision_score(y, probs))
        precision, recall, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)

        return {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "best_threshold": float(threshold)
        }

    def load_champion_model(self) -> Tuple[Optional[Any], str]:
        """Load current champion model from MLflow registry or local disk fallback."""
        champion_model = None
        champion_info = "none"

        # Attempt MLflow load
        try:
            model_uri = f"models:/{self.model_name}@champion"
            champion_model = mlflow.pyfunc.load_model(model_uri)
            champion_info = "mlflow@champion"
        except Exception as e:
            logger.info(f"No MLflow champion alias found ({e}). Checking local heavy_model.pt baseline...")
            local_path = os.path.join(self.model_dir, "heavy_model.pt")
            if os.path.exists(local_path):
                champion_model = HeavyModel.load(local_path)
                champion_info = "local_heavy_model"

        if champion_model is None:
            local_baseline = os.path.join(self.model_dir, "baseline_model.joblib")
            if os.path.exists(local_baseline):
                champion_model = BaselineModel.load(local_baseline)
                champion_info = "local_baseline_model"

        return champion_model, champion_info

    def load_candidate_model(self, candidate_path: Optional[str] = None) -> Tuple[Any, str]:
        """Load candidate model instance from path or default baseline/heavy checkpoint."""
        if candidate_path and os.path.exists(candidate_path):
            if candidate_path.endswith(".pt"):
                model = HeavyModel.load(candidate_path)
            else:
                model = BaselineModel.load(candidate_path)
            return model, candidate_path

        # Default to Heavy Model checkpoint
        heavy_path = os.path.join(self.model_dir, "heavy_model.pt")
        if os.path.exists(heavy_path):
            return HeavyModel.load(heavy_path), "heavy_model.pt"
        
        baseline_path = os.path.join(self.model_dir, "baseline_model.joblib")
        if os.path.exists(baseline_path):
            return BaselineModel.load(baseline_path), "baseline_model.joblib"

        raise FileNotFoundError("No valid candidate model checkpoints found to evaluate.")

    def run_evaluation_and_promotion(
        self,
        candidate_model: Optional[Any] = None,
        champion_model: Optional[Any] = None,
        candidate_name: str = "candidate_model"
    ) -> Dict[str, Any]:
        """
        Main evaluation & gated promotion logic:
        Compares candidate vs champion ROC-AUC on evaluation split.
        Promotes iff Candidate ROC-AUC >= Champion ROC-AUC + delta.
        """
        # Load evaluation test split
        eval_df = load_evaluation_data(self.data_dir)
        feature_cols = [c for c in eval_df.columns if c != "Class"]
        X_eval, y_eval = eval_df[feature_cols], eval_df["Class"]

        # Load models if not passed
        if candidate_model is None:
            candidate_model, candidate_name = self.load_candidate_model()
        if champion_model is None:
            champion_model, champ_info = self.load_champion_model()
        else:
            champ_info = "provided_champion"

        candidate_metrics = self.evaluate_model_on_data(candidate_model, X_eval, y_eval)

        if champion_model is not None:
            champion_metrics = self.evaluate_model_on_data(champion_model, X_eval, y_eval)
        else:
            # Baseline benchmark if no champion exists yet
            champion_metrics = {"roc_auc": 0.50, "pr_auc": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

        candidate_auc = candidate_metrics["roc_auc"]
        champion_auc = champion_metrics["roc_auc"]
        auc_delta = candidate_auc - champion_auc

        promoted = auc_delta >= self.auc_threshold_delta

        report = {
            "candidate_name": candidate_name,
            "champion_info": champ_info,
            "threshold_delta_required": self.auc_threshold_delta,
            "candidate_metrics": candidate_metrics,
            "champion_metrics": champion_metrics,
            "roc_auc_delta": round(auc_delta, 5),
            "promoted": promoted,
            "decision": "PROMOTED_TO_CHAMPION" if promoted else "REJECTED_UNDERPERFORMING"
        }

        # Log decision
        logger.info(f"Candidate ROC-AUC: {candidate_auc:.4f} | Champion ROC-AUC: {champion_auc:.4f} | Delta: {auc_delta:+.4f}")
        if promoted:
            logger.info(f"SUCCESS: Candidate passed promotion gate (Delta {auc_delta:+.4f} >= {self.auc_threshold_delta})! Promoting to Champion.")
            self._promote_in_registry(candidate_name)
        else:
            logger.warning(f"REJECTED: Candidate failed promotion gate (Delta {auc_delta:+.4f} < {self.auc_threshold_delta}). Retaining current Champion.")

        # Save report
        os.makedirs(self.model_dir, exist_ok=True)
        report_path = os.path.join(self.model_dir, "promotion_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        return report

    def _promote_in_registry(self, candidate_name: str):
        """Update MLflow registry alias for champion model if registered model exists."""
        try:
            versions = self.client.search_model_versions(f"name='{self.model_name}'")
            if versions:
                latest_version = versions[0].version
                self.client.set_registered_model_alias(self.model_name, "champion", latest_version)
                logger.info(f"Promoted version {latest_version} of {self.model_name} to alias 'champion'")
        except Exception as e:
            logger.warning(f"MLflow alias promotion skipped or failed ({e}).")


def evaluate_and_promote(
    data_dir: str = "data/processed",
    model_dir: str = "models",
    threshold: float = 0.01,
    candidate_path: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """Helper function for script / test invocations."""
    promoter = ModelEvaluatorPromoter(
        data_dir=data_dir,
        model_dir=model_dir,
        auc_threshold_delta=threshold
    )
    c_model, c_name = promoter.load_candidate_model(candidate_path)
    report = promoter.run_evaluation_and_promotion(candidate_model=c_model, candidate_name=c_name)
    return report["promoted"], report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval-gated Model Promotion Pipeline")
    parser.add_argument("--data-dir", default="data/processed", help="Path to processed data directory")
    parser.add_argument("--model-dir", default="models", help="Path to model directory")
    parser.add_argument("--threshold", type=float, default=0.01, help="ROC-AUC improvement delta required for promotion")
    parser.add_argument("--candidate-path", default=None, help="Path to specific candidate model checkpoint")

    args = parser.parse_args()
    promoted, report = evaluate_and_promote(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        threshold=args.threshold,
        candidate_path=args.candidate_path
    )

    if not promoted:
        logger.error(f"CI/CD Eval Gate Failed: Candidate ROC-AUC delta ({report['roc_auc_delta']:+.4f}) did not meet threshold (+{args.threshold}).")
        sys.exit(1)
    else:
        logger.info(f"CI/CD Eval Gate Passed: Model successfully promoted.")
        sys.exit(0)
