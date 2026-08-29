"""
Statistical Data Drift Detection Module.
Implements Kolmogorov-Smirnov (KS) Test and Population Stability Index (PSI)
to monitor production feature drift against baseline training data.
"""
import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def calculate_psi(reference: np.ndarray, production: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and production feature distributions.
    PSI = sum((P_i - Q_i) * ln(P_i / Q_i))
    """
    ref = np.asarray(reference, dtype=float)
    prod = np.asarray(production, dtype=float)

    if len(ref) == 0 or len(prod) == 0:
        return 0.0

    # Define bin edges using quantiles from reference data
    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(ref, percentiles)
    bins = np.unique(bins)  # Deduplicate identical quantiles

    if len(bins) < 2:
        return 0.0

    # Expand edge boundaries to cover full range
    bins[0] = -np.inf
    bins[-1] = np.inf

    # Calculate bin counts
    ref_counts, _ = np.histogram(ref, bins=bins)
    prod_counts, _ = np.histogram(prod, bins=bins)

    # Convert counts to proportions with epsilon smoothing to prevent zero division / log(0)
    eps = 1e-4
    ref_props = (ref_counts + eps) / (len(ref) + eps * len(ref_counts))
    prod_props = (prod_counts + eps) / (len(prod) + eps * len(prod_counts))

    # Compute PSI sum
    psi_value = np.sum((prod_props - ref_props) * np.log(prod_props / ref_props))
    return float(psi_value)


class DriftDetector:
    """Monitors feature distribution drift using rolling window buffers vs baseline dataset."""

    def __init__(self, data_dir: str = "data/processed", window_capacity: int = 1000):
        self.data_dir = data_dir
        self.window_capacity = window_capacity
        self.feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        self.reference_data: Optional[pd.DataFrame] = None
        self.production_buffer: List[Dict[str, float]] = []

        self._load_reference_data()

    def _load_reference_data(self):
        """Load reference baseline training dataset."""
        train_path = os.path.join(self.data_dir, "train.parquet")
        if not os.path.exists(train_path):
            train_path = os.path.join(self.data_dir, "train.csv")

        if os.path.exists(train_path):
            df = pd.read_parquet(train_path) if train_path.endswith(".parquet") else pd.read_csv(train_path)
            self.reference_data = df[[c for c in self.feature_cols if c in df.columns]]
            logger.info(f"Loaded reference dataset with shape {self.reference_data.shape} for drift monitoring.")
        else:
            logger.warning(f"Reference dataset not found at {train_path}. Generating synthetic reference buffer.")
            dummy_matrix = np.random.normal(0, 1, size=(500, len(self.feature_cols)))
            self.reference_data = pd.DataFrame(dummy_matrix, columns=self.feature_cols)

    def add_production_sample(self, sample: Dict[str, float]):
        """Add a incoming production sample to the rolling window buffer."""
        self.production_buffer.append(sample)
        if len(self.production_buffer) > self.window_capacity:
            self.production_buffer.pop(0)

    def add_production_batch(self, batch: List[Dict[str, float]]):
        """Add a batch of incoming samples to the rolling window buffer."""
        for sample in batch:
            self.add_production_sample(sample)

    def compute_drift_report(self) -> Dict[str, Any]:
        """
        Compute KS statistic, KS p-value, and PSI score for all features.
        Returns detailed feature-level and summary drift report.
        """
        if not self.production_buffer or self.reference_data is None:
            return {
                "status": "insufficient_data",
                "sample_count": len(self.production_buffer),
                "features": {}
            }

        prod_df = pd.DataFrame(self.production_buffer)
        feature_reports = {}
        max_psi = 0.0
        drift_alert_features = []

        for col in self.feature_cols:
            if col in prod_df.columns and col in self.reference_data.columns:
                ref_vals = self.reference_data[col].dropna().values
                prod_vals = prod_df[col].dropna().values

                if len(prod_vals) < 5:
                    continue

                # KS test
                ks_stat, p_val = stats.ks_2samp(ref_vals, prod_vals)

                # PSI calculation
                psi_val = calculate_psi(ref_vals, prod_vals, num_bins=10)

                if psi_val > max_psi:
                    max_psi = psi_val

                # Alert threshold: PSI >= 0.25 or KS p-value < 0.01
                has_drift = bool(psi_val >= 0.25 or (p_val < 0.01 and len(prod_vals) >= 50))
                if has_drift:
                    drift_alert_features.append(col)

                feature_reports[col] = {
                    "ks_stat": round(float(ks_stat), 4),
                    "ks_pvalue": round(float(p_val), 4),
                    "psi": round(float(psi_val), 4),
                    "drift_detected": has_drift
                }

        summary_status = "significant_drift" if len(drift_alert_features) >= 3 or max_psi >= 0.25 else (
            "moderate_drift" if max_psi >= 0.10 else "no_drift"
        )

        return {
            "status": summary_status,
            "sample_count": len(self.production_buffer),
            "max_psi": round(float(max_psi), 4),
            "drift_alert_features": drift_alert_features,
            "features": feature_reports
        }
