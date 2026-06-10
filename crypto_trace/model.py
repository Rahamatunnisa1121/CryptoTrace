"""
model.py
────────
Step 3 of the ML pipeline — Train final RF classifier on 15 features,
serialize model + pipeline, and expose compute_dirty_score() for live use.

Key design:
  • Schema alignment is enforced here: live wallet features MUST use the
    exact same 15 column names as the Kaggle training set.
  • compute_dirty_score() returns predict_proba(fraud_class) in [0, 1].

Usage:
    python -m crypto_trace.model          # train & save
"""
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_AVAILABLE = True
except ImportError:
    _SMOTE_AVAILABLE = False
    SMOTE = None
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (
    DATA_PROCESSED,
    DIRTY_SCORE_HIGH,
    DIRTY_SCORE_MED,
    FEATURES_CONFIG_PATH,
    MODEL_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    TOP_K_FEATURES,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def load_features_config() -> list[str]:
    """Load the 15 selected feature names from JSON config."""
    if not FEATURES_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Feature config not found at {FEATURES_CONFIG_PATH}. "
            "Run feature_selection.py first."
        )
    with open(FEATURES_CONFIG_PATH) as f:
        return json.load(f)


def risk_label(score: float) -> str:
    """Convert numeric dirty score to human-readable risk label."""
    if score >= DIRTY_SCORE_HIGH:
        return "HIGH"
    if score >= DIRTY_SCORE_MED:
        return "MEDIUM"
    return "LOW"


# ─── Training ────────────────────────────────────────────────────────────────

def load_training_data(top_features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Load the 15-feature reduced dataset."""
    path = DATA_PROCESSED / "optimized_15_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run feature_selection.py first. Missing: {path}")
    df = pd.read_csv(path)
    X = df[top_features]
    y = df["FLAG"]
    log.info(f"Loaded training data: {X.shape} | fraud rate: {y.mean():.2%}")
    return X, y


def upsample_with_smote(
    X: pd.DataFrame, y: pd.Series
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to address 77/23 class imbalance.
    SMOTE synthesizes new minority (fraud) samples rather than duplicating,
    giving the model a balanced 50/50 view during training.

    Falls back to class_weight='balanced' on the RF if imblearn is not installed.
    """
    if not _SMOTE_AVAILABLE:
        log.warning(
            "imbalanced-learn not installed. Skipping SMOTE — "
            "class_weight='balanced' on the RF will compensate."
        )
        return X.values if hasattr(X, "values") else X, y.values if hasattr(y, "values") else y

    log.info("Applying SMOTE upsampling to balance classes…")
    sm = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X, y)
    log.info(
        f"After SMOTE → {len(X_res)} samples | "
        f"fraud rate: {y_res.mean():.2%}"
    )
    return X_res, y_res


def train_final_pipeline(
    X: pd.DataFrame, y: pd.Series, top_features: list[str]
) -> Pipeline:
    """
    Build and train the final sklearn Pipeline:
        StandardScaler → RandomForestClassifier (on 15 features)

    Uses SMOTE-upsampled data for training.
    """
    X_res, y_res = upsample_with_smote(X, y)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=N_ESTIMATORS,
                    class_weight="balanced",
                    max_depth=20,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # Hold-out validation
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, stratify=y_res, random_state=RANDOM_STATE
    )
    log.info("Training final RF pipeline on 15 features…")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    log.info("\n" + classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    log.info(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    return pipeline


def save_model(pipeline: Pipeline) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    log.info(f"Saved final model pipeline → {MODEL_PATH}")


# ─── Inference (used at runtime by graph_builder) ────────────────────────────

class DirtyScorer:
    """
    Thin wrapper around the trained sklearn Pipeline.
    Loaded once at startup; compute_dirty_score() is called per wallet.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._features: list[str] | None = None

    def load(self) -> None:
        """Load model + feature config from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python -m crypto_trace.model"
            )
        self._pipeline = joblib.load(MODEL_PATH)
        self._features = load_features_config()
        log.info(f"DirtyScorer loaded. Features: {self._features}")

    @property
    def features(self) -> list[str]:
        if self._features is None:
            raise RuntimeError("DirtyScorer not loaded. Call .load() first.")
        return self._features

    def compute_dirty_score(self, feature_dict: dict[str, float]) -> dict[str, Any]:
        """
        Score a single wallet.

        Args:
            feature_dict: exactly the 15 feature keys → float values
        Returns:
            {
                "dirtyScore": float [0, 1],
                "riskLevel":  "LOW" | "MEDIUM" | "HIGH",
            }
        """
        if self._pipeline is None:
            raise RuntimeError(
                "DirtyScorer not loaded. Call .load() first."
            )

        # Build a single-row DataFrame with correct column order
        row = {f: feature_dict.get(f, 0.0) for f in self.features}
        X_live = pd.DataFrame([row])

        score: float = float(self._pipeline.predict_proba(X_live)[0, 1])
        return {
            "dirtyScore": round(score, 4),
            "riskLevel": risk_label(score),
        }
    """
    Thin wrapper around the trained sklearn Pipeline.
    Loaded once at startup; compute_dirty_score() is called per wallet.
    """

    def __init__(self) -> None:
        self._pipeline: Pipeline | None = None
        self._features: list[str] | None = None

    def load(self) -> None:
        """Load model + feature config from disk."""
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python -m crypto_trace.model"
            )
        self._pipeline = joblib.load(MODEL_PATH)
        self._features = load_features_config()
        log.info(f"DirtyScorer loaded. Features: {self._features}")

    @property
    def features(self) -> list[str]:
        if self._features is None:
            raise RuntimeError("DirtyScorer not loaded. Call .load() first.")
        return self._features

def compute_dirty_score(self, feature_dict: dict[str, float]) -> dict[str, Any]:
        if self._pipeline is None:
            raise RuntimeError(
                "DirtyScorer not loaded. Call .load() first."
            )

        # Build a single-row DataFrame with correct column order
        row = {f: feature_dict.get(f, 0.0) for f in self.features}
        X_live = pd.DataFrame([row])

        score: float = float(self._pipeline.predict_proba(X_live)[0, 1])
        return {
            "dirtyScore": round(score, 4),
            "riskLevel": risk_label(score),
        }


# ─── Singleton ────────────────────────────────────────────────────────────────
scorer = DirtyScorer()


# ─── Entry-point ─────────────────────────────────────────────────────────────

def run() -> None:
    top_features = load_features_config()
    X, y = load_training_data(top_features)
    pipeline = train_final_pipeline(X, y, top_features)
    save_model(pipeline)


if __name__ == "__main__":
    run()
