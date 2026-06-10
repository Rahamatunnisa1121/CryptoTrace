"""
tests/test_data_prep.py
────────────────────────
Unit tests for the data cleaning and preprocessing pipeline.

Run:
    pytest tests/test_data_prep.py -v
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crypto_trace.data_prep import clean, build_preprocessing_pipeline, NON_PREDICTIVE_COLS, TARGET_COL


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """Minimal synthetic DataFrame that mirrors the Kaggle schema."""
    rng = np.random.default_rng(42)
    n = 100
    data = {
        "": range(n),
        "Index": range(1, n + 1),
        "Address": [f"0x{'a' * 40}" for _ in range(n)],
        "FLAG": rng.integers(0, 2, n),
        "Avg min between sent tnx": rng.uniform(0, 10000, n),
        "Avg min between received tnx": rng.uniform(0, 10000, n),
        "Time Diff between first and last (Mins)": rng.uniform(0, 1_500_000, n),
        "Sent tnx": rng.integers(0, 500, n).astype(float),
        "Received Tnx": rng.integers(0, 500, n).astype(float),
        "total Ether sent": rng.uniform(0, 1000, n),
        "total ether received": rng.uniform(0, 1000, n),
        "total ether balance": rng.uniform(-500, 500, n),
        "ERC20 most sent token type": ["TokenA"] * n,
        "ERC20_most_rec_token_type": ["TokenB"] * n,
    }
    # Introduce some NaN values to test imputation
    df = pd.DataFrame(data)
    df.loc[5:10, "Sent tnx"] = np.nan
    df.loc[20:25, "total Ether sent"] = np.nan
    return df


# ─── Tests ───────────────────────────────────────────────────────────────────

class TestClean:
    def test_drops_non_predictive_columns(self, sample_df):
        X, y = clean(sample_df)
        for col in NON_PREDICTIVE_COLS:
            assert col not in X.columns, f"Column '{col}' should have been dropped"

    def test_target_not_in_X(self, sample_df):
        X, y = clean(sample_df)
        assert TARGET_COL not in X.columns

    def test_y_is_binary(self, sample_df):
        _, y = clean(sample_df)
        assert set(y.unique()).issubset({0, 1})

    def test_no_nan_after_imputation(self, sample_df):
        X, _ = clean(sample_df)
        assert X.isna().sum().sum() == 0, "No NaN values should remain after cleaning"

    def test_only_numeric_columns(self, sample_df):
        X, _ = clean(sample_df)
        for dtype in X.dtypes:
            assert np.issubdtype(dtype, np.number), "All columns should be numeric"

    def test_shape_reasonable(self, sample_df):
        X, y = clean(sample_df)
        assert len(X) == len(sample_df)
        assert len(y) == len(sample_df)
        assert X.shape[1] < sample_df.shape[1]  # some columns dropped


class TestPreprocessingPipeline:
    def test_pipeline_transforms_correctly(self, sample_df):
        X, _ = clean(sample_df)
        pipeline = build_preprocessing_pipeline(X)
        X_scaled = pipeline.transform(X)

        # After StandardScaler: mean ≈ 0, std ≈ 1 per column
        means = X_scaled.mean(axis=0)
        stds = X_scaled.std(axis=0)
        assert np.allclose(means, 0, atol=1e-10), "Scaled means should be ~0"
        assert np.allclose(stds, 1, atol=0.1), "Scaled stds should be ~1"

    def test_pipeline_output_shape(self, sample_df):
        X, _ = clean(sample_df)
        pipeline = build_preprocessing_pipeline(X)
        X_scaled = pipeline.transform(X)
        assert X_scaled.shape == X.shape

    def test_pipeline_is_serializable(self, sample_df, tmp_path):
        import joblib
        X, _ = clean(sample_df)
        pipeline = build_preprocessing_pipeline(X)
        path = tmp_path / "test_pipeline.pkl"
        joblib.dump(pipeline, path)
        loaded = joblib.load(path)
        # Both should produce identical output
        out1 = pipeline.transform(X)
        out2 = loaded.transform(X)
        assert np.allclose(out1, out2)
