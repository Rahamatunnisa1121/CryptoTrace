"""
data_prep.py
────────────
Step 1 of the ML pipeline:
  - Load raw Kaggle Ethereum Fraud Detection dataset
  - Remove non-predictive columns (Address, Index)
  - Impute missing values with 0 (standard for this dataset)
  - Scale numeric features with StandardScaler
  - Save cleaned dataset + reusable sklearn Pipeline

Usage:
    python -m crypto_trace.data_prep
"""
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DATA_PROCESSED, DATA_RAW, RANDOM_STATE, TOP_K_FEATURES

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Columns to drop before any ML step ─────────────────────────────────────
NON_PREDICTIVE_COLS = [
    "",           # unnamed index col from CSV
    "Index",
    "Address",
    "ERC20 most sent token type",
    "ERC20_most_rec_token_type",
]

TARGET_COL = "FLAG"


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load the raw Kaggle CSV."""
    path = path or DATA_RAW / "transaction_dataset.csv"
    log.info(f"Loading raw dataset from {path}")
    df = pd.read_csv(path, index_col=0)
    log.info(f"Shape: {df.shape}")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop non-predictive columns, impute NaN → 0, split X / y.

    Returns:
        X  (DataFrame) – feature matrix, raw numeric values
        y  (Series)    – binary label (0 = normal, 1 = fraud)
    """
    # Drop non-predictive cols that exist in the frame
    to_drop = [c for c in NON_PREDICTIVE_COLS if c in df.columns]
    log.info(f"Dropping columns: {to_drop}")
    df = df.drop(columns=to_drop, errors="ignore")

    # Separate target
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # Keep only numeric columns (drop token-name strings etc.)
    X = X.select_dtypes(include=[np.number])

    # Fill missing values with 0 (documented choice: no leakage risk)
    nan_count = X.isna().sum().sum()
    log.info(f"Imputing {nan_count} missing values with 0")
    X = X.fillna(0)

    log.info(f"Clean X shape: {X.shape} | y fraud rate: {y.mean():.2%}")
    return X, y


def build_preprocessing_pipeline(X: pd.DataFrame) -> Pipeline:
    """
    Build a simple sklearn Pipeline:
        StandardScaler (zero mean, unit variance)

    Fitting on the DataFrame (not .values) preserves feature names so that
    downstream predict calls with named DataFrames don't trigger sklearn warnings.
    """
    pipeline = Pipeline([("scaler", StandardScaler())])
    pipeline.fit(X)  # X is a DataFrame → scaler stores feature_names_in_
    return pipeline


def save_artifacts(
    X: pd.DataFrame,
    y: pd.Series,
    pipeline: Pipeline,
    feature_names: list[str],
) -> None:
    """Persist processed dataset and pipeline to data/processed/."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Scaled feature matrix
    X_scaled = pd.DataFrame(
        pipeline.transform(X), columns=feature_names, index=X.index
    )
    X_scaled[TARGET_COL] = y.values
    out_path = DATA_PROCESSED / "optimized_transaction_dataset.csv"
    X_scaled.to_csv(out_path, index=False)
    log.info(f"Saved cleaned+scaled dataset → {out_path}")

    # Pipeline
    pipeline_path = DATA_PROCESSED / "preprocessing_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    log.info(f"Saved preprocessing pipeline → {pipeline_path}")

    # Feature list (for downstream reference)
    features_path = DATA_PROCESSED / "all_features.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    log.info(f"Saved feature list → {features_path}")


def run() -> tuple[pd.DataFrame, pd.Series, Pipeline, list[str]]:
    """Full data prep pipeline. Returns X, y, pipeline, feature_names."""
    df = load_raw()
    X, y = clean(df)
    pipeline = build_preprocessing_pipeline(X)
    feature_names = list(X.columns)
    save_artifacts(X, y, pipeline, feature_names)
    return X, y, pipeline, feature_names


if __name__ == "__main__":
    run()
