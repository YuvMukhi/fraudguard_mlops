"""
Lightweight Baseline Fraud Detection Model.
Uses Class-Weighted Logistic Regression with Precision-Recall Threshold Tuning.
"""
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score, auc
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BaselineModel:
    """Baseline Logistic Regression Model with class weighting and threshold tuning."""

    def __init__(self, C: float = 1.0, class_weight: str = "balanced", max_iter: int = 1000, random_state: int = 42):
        self.C = C
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs"
        )
        self.best_threshold: float = 0.5

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> "BaselineModel":
        """Fit baseline model and optimize decision threshold on validation set if provided."""
        logger.info(f"Training Baseline Model (LogisticRegression, C={self.C}, class_weight={self.class_weight})...")
        self.model.fit(X_train, y_train)

        if X_val is not None and y_val is not None:
            self.best_threshold = self.optimize_threshold(X_val, y_val)
            logger.info(f"Baseline optimal threshold set to: {self.best_threshold:.4f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probability estimates (positive class prob)."""
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Predict binary class labels using specified or tuned threshold."""
        thresh = threshold if threshold is not None else self.best_threshold
        probs = self.predict_proba(X)
        return (probs >= thresh).astype(int)

    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Find decision threshold that maximizes F1-score on validation set."""
        probs = self.predict_proba(X_val)
        precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
        
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
        best_thresh = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        return best_thresh

    def evaluate(self, X: pd.DataFrame, y: pd.Series, threshold: float = None) -> Dict[str, float]:
        """Compute performance metrics (PR-AUC, ROC-AUC, Precision, Recall, F1)."""
        probs = self.predict_proba(X)
        preds = self.predict(X, threshold=threshold)
        
        precisions, recalls, _ = precision_recall_curve(y, probs)
        pr_auc = auc(recalls, precisions)
        roc_auc = roc_auc_score(y, probs)

        return {
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1_score": float(f1_score(y, preds, zero_division=0)),
            "threshold": float(threshold if threshold is not None else self.best_threshold)
        }

    def save(self, filepath: str):
        """Save model checkpoint to disk."""
        joblib.dump({"model": self.model, "best_threshold": self.best_threshold}, filepath)

    @classmethod
    def load(cls, filepath: str) -> "BaselineModel":
        """Load model checkpoint from disk."""
        data = joblib.load(filepath)
        instance = cls()
        instance.model = data["model"]
        instance.best_threshold = data["best_threshold"]
        return instance
