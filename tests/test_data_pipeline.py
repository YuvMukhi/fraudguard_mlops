"""
Unit tests for data pipeline and synthetic fraud dataset generator.
"""
import os
import shutil
import pytest
import pandas as pd
import numpy as np
from src.data.pipeline import FraudDataGenerator, DataPipeline


@pytest.fixture
def temp_output_dir(tmp_path):
    d = tmp_path / "processed_data"
    d.mkdir()
    return str(d)


def test_fraud_data_generator():
    n_samples = 1000
    positive_rate = 0.01  # 1% for small test
    generator = FraudDataGenerator(n_samples=n_samples, positive_rate=positive_rate, random_state=42)
    df = generator.generate()

    assert len(df) == n_samples
    assert "Class" in df.columns
    assert "Time" in df.columns
    assert "Amount" in df.columns
    for i in range(1, 29):
        assert f"V{i}" in df.columns

    fraud_count = df["Class"].sum()
    assert fraud_count == int(n_samples * positive_rate)


def test_data_pipeline_cleaning():
    pipeline = DataPipeline()
    generator = FraudDataGenerator(n_samples=500, positive_rate=0.02, random_state=42)
    raw_df = generator.generate()

    # Introduce synthetic null and duplicate
    raw_df.loc[0, "V1"] = np.nan
    dup_row = raw_df.iloc[1:2].copy()
    df_with_issues = pd.concat([raw_df, dup_row], axis=0)

    clean_df = pipeline.clean_data(df_with_issues)

    assert clean_df["V1"].isnull().sum() == 0
    assert len(clean_df) <= len(raw_df) - 1


def test_stratified_split_and_scaling(temp_output_dir):
    pipeline = DataPipeline(random_state=42)
    generator = FraudDataGenerator(n_samples=2000, positive_rate=0.01, random_state=42)
    df = generator.generate()

    train_df, val_df, test_df, stats = pipeline.preprocess_and_split(df, train_size=0.7, val_size=0.15, test_size=0.15)

    # Check total row count matches
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Check stratification: fraud rates should be close across splits
    overall_rate = df["Class"].mean()
    assert abs(stats["train_fraud_rate"] - overall_rate) < 0.005
    assert abs(stats["val_fraud_rate"] - overall_rate) < 0.005
    assert abs(stats["test_fraud_rate"] - overall_rate) < 0.005

    # Check scaled features mean ~0 and std ~1 for train set
    feature_cols = [c for c in train_df.columns if c != "Class"]
    train_means = train_df[feature_cols].mean()
    train_stds = train_df[feature_cols].std()

    assert np.allclose(train_means, 0.0, atol=1e-1)
    assert np.allclose(train_stds, 1.0, atol=1e-1)


def test_run_pipeline_end_to_end(temp_output_dir):
    pipeline = DataPipeline(random_state=42)
    metadata = pipeline.run_pipeline(
        output_dir=temp_output_dir,
        n_samples=1000,
        positive_rate=0.01
    )

    assert os.path.exists(os.path.join(temp_output_dir, "train.parquet"))
    assert os.path.exists(os.path.join(temp_output_dir, "val.parquet"))
    assert os.path.exists(os.path.join(temp_output_dir, "test.parquet"))
    assert os.path.exists(os.path.join(temp_output_dir, "scaler.joblib"))
    assert os.path.exists(os.path.join(temp_output_dir, "metadata.json"))

    assert "stats" in metadata
    assert metadata["stats"]["train_shape"][0] > 0
