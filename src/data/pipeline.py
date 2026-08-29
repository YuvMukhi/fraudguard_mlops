"""
Data Ingestion, Synthetic Generation, Preprocessing, and Stratified Pipeline.
Mirrors the Kaggle Credit Card Fraud Detection dataset (~0.2% positive fraud rate).
"""
import os
import json
import logging
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FraudDataGenerator:
    """Generates synthetic Credit Card Fraud Detection dataset with realistic imbalance."""

    def __init__(self, n_samples: int = 50000, positive_rate: float = 0.002, random_state: int = 42):
        self.n_samples = n_samples
        self.positive_rate = positive_rate
        self.random_state = random_state

    def generate(self) -> pd.DataFrame:
        """Generate synthetic DataFrame matching Kaggle Credit Card Fraud schema."""
        np.random.seed(self.random_state)
        n_fraud = max(1, int(self.n_samples * self.positive_rate))
        n_genuine = self.n_samples - n_fraud

        logger.info(f"Generating synthetic dataset: {n_genuine} genuine, {n_fraud} fraud samples ({self.positive_rate*100:.2f}% positive rate)")

        # Features V1-V28 (PCA components)
        # Genuine: centered around 0 with unit variance
        # Fraud: distinct mean shifts and higher variance on key features (e.g., V4, V11, V14)
        v_genuine = np.random.normal(loc=0.0, scale=1.0, size=(n_genuine, 28))
        
        v_fraud = np.random.normal(loc=0.0, scale=1.2, size=(n_fraud, 28))
        # Add realistic fraud signals to specific PCA features
        v_fraud[:, 3] += 2.5   # V4 shift
        v_fraud[:, 10] += 3.0  # V11 shift
        v_fraud[:, 13] -= 3.5  # V14 shift
        v_fraud[:, 16] -= 2.0  # V17 shift

        # Time feature: simulated seconds over 2 days (0 to 172800 seconds)
        time_genuine = np.sort(np.random.uniform(0, 172800, size=n_genuine))
        time_fraud = np.random.uniform(0, 172800, size=n_fraud)

        # Amount feature: log-normal distribution
        amount_genuine = np.random.lognormal(mean=3.0, sigma=1.0, size=n_genuine)
        amount_fraud = np.random.lognormal(mean=4.5, sigma=1.5, size=n_fraud)

        # Combine
        data_genuine = np.column_stack((time_genuine, v_genuine, amount_genuine))
        data_fraud = np.column_stack((time_fraud, v_fraud, amount_fraud))

        cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        
        df_genuine = pd.DataFrame(data_genuine, columns=cols)
        df_genuine["Class"] = 0

        df_fraud = pd.DataFrame(data_fraud, columns=cols)
        df_fraud["Class"] = 1

        df = pd.concat([df_genuine, df_fraud], axis=0).sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        return df


class DataPipeline:
    """Pipeline for data loading, cleaning, stratified splitting, and feature scaling."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        self.target_col = "Class"

    def load_or_generate_data(self, file_path: Optional[str] = None, n_samples: int = 50000, positive_rate: float = 0.002) -> pd.DataFrame:
        """Load data from CSV file or generate synthetic equivalent if file does not exist."""
        if file_path and os.path.exists(file_path):
            logger.info(f"Loading dataset from file: {file_path}")
            df = pd.read_csv(file_path)
        else:
            logger.info("Dataset file not found or not specified. Generating synthetic fraud dataset.")
            generator = FraudDataGenerator(n_samples=n_samples, positive_rate=positive_rate, random_state=self.random_state)
            df = generator.generate()
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean dataset by handling null values and validating schema."""
        initial_len = len(df)
        df = df.dropna().drop_duplicates().reset_index(drop=True)
        cleaned_len = len(df)
        logger.info(f"Cleaned dataset: dropped {initial_len - cleaned_len} rows (nulls/duplicates). Final rows: {cleaned_len}")
        return df

    def preprocess_and_split(
        self,
        df: pd.DataFrame,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Perform stratified train/val/test split and scale features without data leakage.
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-5, "Split ratios must sum to 1.0"

        X = df[self.feature_cols].copy()
        y = df[self.target_col].copy()

        # Step 1: Separate test set stratifiably
        test_ratio = test_size
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=self.random_state
        )

        # Step 2: Separate train and val sets from remaining temp data
        val_ratio_relative = val_size / (train_size + val_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio_relative, stratify=y_temp, random_state=self.random_state
        )

        # Fit scaler ONLY on train set to prevent data leakage
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), columns=self.feature_cols, index=X_train.index)
        X_val_scaled = pd.DataFrame(self.scaler.transform(X_val), columns=self.feature_cols, index=X_val.index)
        X_test_scaled = pd.DataFrame(self.scaler.transform(X_test), columns=self.feature_cols, index=X_test.index)

        # Combine features with target
        train_df = pd.concat([X_train_scaled, y_train], axis=1)
        val_df = pd.concat([X_val_scaled, y_val], axis=1)
        test_df = pd.concat([X_test_scaled, y_test], axis=1)

        stats = {
            "train_shape": train_df.shape,
            "train_fraud_count": int(y_train.sum()),
            "train_fraud_rate": float(y_train.mean()),
            "val_shape": val_df.shape,
            "val_fraud_count": int(y_val.sum()),
            "val_fraud_rate": float(y_val.mean()),
            "test_shape": test_df.shape,
            "test_fraud_count": int(y_test.sum()),
            "test_fraud_rate": float(y_test.mean()),
        }

        logger.info(f"Train set: {stats['train_shape']} (Fraud rate: {stats['train_fraud_rate']*100:.3f}%)")
        logger.info(f"Val set:   {stats['val_shape']} (Fraud rate: {stats['val_fraud_rate']*100:.3f}%)")
        logger.info(f"Test set:  {stats['test_shape']} (Fraud rate: {stats['test_fraud_rate']*100:.3f}%)")

        return train_df, val_df, test_df, stats

    def run_pipeline(
        self,
        output_dir: str = "data/processed",
        file_path: Optional[str] = None,
        n_samples: int = 50000,
        positive_rate: float = 0.002
    ) -> Dict[str, Any]:
        """Execute complete data pipeline and save artifacts."""
        os.makedirs(output_dir, exist_ok=True)
        raw_df = self.load_or_generate_data(file_path=file_path, n_samples=n_samples, positive_rate=positive_rate)
        clean_df = self.clean_data(raw_df)
        train_df, val_df, test_df, stats = self.preprocess_and_split(clean_df)

        # Save datasets as parquet and csv for compatibility
        train_df.to_parquet(os.path.join(output_dir, "train.parquet"), index=False)
        val_df.to_parquet(os.path.join(output_dir, "val.parquet"), index=False)
        test_df.to_parquet(os.path.join(output_dir, "test.parquet"), index=False)

        train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
        test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

        # Save scaler artifact
        scaler_path = os.path.join(output_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        # Save metadata
        metadata = {
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "stats": stats,
            "scaler_path": scaler_path
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Pipeline executed successfully. Artifacts saved to: {output_dir}")
        return metadata


if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run_pipeline()
