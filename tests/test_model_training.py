"""
Unit tests for Baseline and Heavy model training and evaluation.
"""
import os
import pytest
import pandas as pd
import numpy as np
from src.data.pipeline import FraudDataGenerator, DataPipeline
from src.models.baseline import BaselineModel
from src.models.heavy import HeavyModel


@pytest.fixture
def sample_dataset(tmp_path):
    generator = FraudDataGenerator(n_samples=1000, positive_rate=0.02, random_state=42)
    df = generator.generate()
    pipeline = DataPipeline(random_state=42)
    train_df, val_df, test_df, _ = pipeline.preprocess_and_split(df, train_size=0.7, val_size=0.15, test_size=0.15)
    return train_df, val_df, test_df


def test_baseline_model_training(sample_dataset, tmp_path):
    train_df, val_df, test_df = sample_dataset
    feature_cols = [c for c in train_df.columns if c != "Class"]
    
    baseline = BaselineModel(C=1.0, class_weight="balanced", random_state=42)
    baseline.fit(train_df[feature_cols], train_df["Class"], val_df[feature_cols], val_df["Class"])

    # Test predictions
    probs = baseline.predict_proba(test_df[feature_cols])
    preds = baseline.predict(test_df[feature_cols])

    assert len(probs) == len(test_df)
    assert len(preds) == len(test_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert set(preds).issubset({0, 1})

    # Test evaluation metrics
    metrics = baseline.evaluate(test_df[feature_cols], test_df["Class"])
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "f1_score" in metrics
    assert metrics["pr_auc"] > 0.0

    # Test saving and loading checkpoint
    save_path = str(tmp_path / "baseline.joblib")
    baseline.save(save_path)
    loaded_baseline = BaselineModel.load(save_path)
    
    loaded_probs = loaded_baseline.predict_proba(test_df[feature_cols])
    assert np.allclose(probs, loaded_probs)


def test_heavy_model_training(sample_dataset, tmp_path):
    train_df, val_df, test_df = sample_dataset
    feature_cols = [c for c in train_df.columns if c != "Class"]

    heavy = HeavyModel(input_dim=len(feature_cols), epochs=3, batch_size=64, pos_weight=10.0, random_state=42)
    heavy.fit(train_df[feature_cols], train_df["Class"], val_df[feature_cols], val_df["Class"])

    # Test predictions
    probs = heavy.predict_proba(test_df[feature_cols])
    preds = heavy.predict(test_df[feature_cols])

    assert len(probs) == len(test_df)
    assert len(preds) == len(test_df)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    assert set(preds).issubset({0, 1})

    # Test evaluation metrics
    metrics = heavy.evaluate(test_df[feature_cols], test_df["Class"])
    assert "pr_auc" in metrics
    assert "roc_auc" in metrics
    assert "f1_score" in metrics

    # Test saving and loading checkpoint
    save_path = str(tmp_path / "heavy.pt")
    heavy.save(save_path)
    loaded_heavy = HeavyModel.load(save_path)

    loaded_probs = loaded_heavy.predict_proba(test_df[feature_cols])
    assert np.allclose(probs, loaded_probs, atol=1e-4)
