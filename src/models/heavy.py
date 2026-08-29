"""
Heavy Fraud Detection Model.
PyTorch Deep Neural Network with weighted BCE loss, Dropout, and Threshold Optimization.
"""
import logging
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_curve, roc_auc_score, precision_score, recall_score, f1_score, auc
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PyTorchFraudNN(nn.Module):
    """Deep Neural Network architecture for fraud detection."""

    def __init__(self, input_dim: int = 30, hidden_dim1: int = 64, hidden_dim2: int = 32, dropout_rate: float = 0.2):
        super(PyTorchFraudNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim2, 1)  # Logits output for BCEWithLogitsLoss
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HeavyModel:
    """Wrapper for PyTorch Heavy Model with class weighting and threshold tuning."""

    def __init__(
        self,
        input_dim: int = 30,
        epochs: int = 15,
        batch_size: int = 256,
        lr: float = 1e-3,
        pos_weight: float = 50.0,
        random_state: int = 42
    ):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.pos_weight = pos_weight
        self.random_state = random_state

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.model = PyTorchFraudNN(input_dim=input_dim)
        self.best_threshold: float = 0.5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> "HeavyModel":
        """Train PyTorch neural network model with positive class weight."""
        logger.info(f"Training Heavy Model (PyTorch DNN on {self.device}, epochs={self.epochs}, pos_weight={self.pos_weight})...")
        
        # Calculate positive class weight dynamically if pos_weight is default
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        weight_val = self.pos_weight if self.pos_weight > 0 else (n_neg / max(1, n_pos))

        X_t = torch.tensor(X_train.values, dtype=torch.float32)
        y_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight_val], device=self.device))
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_x.size(0)

            avg_loss = epoch_loss / len(dataset)
            if epoch % 5 == 0 or epoch == self.epochs:
                logger.info(f"Epoch {epoch}/{self.epochs} - Loss: {avg_loss:.4f}")

        if X_val is not None and y_val is not None:
            self.best_threshold = self.optimize_threshold(X_val, y_val)
            logger.info(f"Heavy model optimal threshold set to: {self.best_threshold:.4f}")

        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities."""
        self.model.eval()
        X_t = torch.tensor(X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        return probs

    def predict(self, X: pd.DataFrame, threshold: float = None) -> np.ndarray:
        """Predict binary labels using specified or tuned threshold."""
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
        """Compute evaluation metrics."""
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
        """Save PyTorch state dict and parameters."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "best_threshold": self.best_threshold
        }
        torch.save(checkpoint, filepath)

    @classmethod
    def load(cls, filepath: str) -> "HeavyModel":
        """Load PyTorch checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
        instance = cls(input_dim=checkpoint["input_dim"])
        instance.model.load_state_dict(checkpoint["state_dict"])
        instance.best_threshold = checkpoint["best_threshold"]
        instance.model.eval()
        return instance
