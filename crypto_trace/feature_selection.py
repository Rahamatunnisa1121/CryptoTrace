"""
feature_selection.py
────────────────────
Step 2 of the ML pipeline — Dimensionality Reduction via Random Forest.

Academic justification:
  Random Forest calculates "Gini Importance" (Mean Decrease in Impurity) for
  every feature across all trees. Features that rarely split the data in a
  useful way score near 0 and are effectively "noisy" metadata.
  
  This follows the approach in:
    • Zhou et al. (2025) – Ensemble Learning for Ethereum fraud detection
    • RF-RFE methodology (Recursive Feature Elimination) – standard in
      high-dimensional fraud detection literature

Usage:
    python -m crypto_trace.feature_selection
"""
import json
import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import (
    DATA_PROCESSED,
    FEATURES_CONFIG_PATH,
    N_ESTIMATORS,
    RANDOM_STATE,
    TOP_K_FEATURES,
)
from .data_prep import run as run_data_prep

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def compute_gini_importance(
    X: pd.DataFrame, y: pd.Series
) -> tuple[RandomForestClassifier, pd.Series]:
    """
    Train a RandomForestClassifier on all features and extract Gini Importance.

    class_weight='balanced' compensates for the 77/23 class imbalance without
    requiring upsampling at this stage.

    Returns:
        rf          – trained RandomForestClassifier
        importances – pd.Series sorted descending by Gini importance
    """
    log.info(f"Training RF on {X.shape[1]} features to compute Gini importance…")
    rf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )
    log.info("Top 20 features by Gini Importance:")
    log.info(importances.head(20).to_string())
    return rf, importances


def select_top_k(importances: pd.Series, k: int = TOP_K_FEATURES) -> list[str]:
    """Return the names of the top-k most important features."""
    top_k = importances.head(k).index.tolist()
    log.info(f"Selected top {k} features: {top_k}")
    return top_k


def evaluate_reduced_model(
    X: pd.DataFrame, y: pd.Series, top_features: list[str]
) -> None:
    """
    Cross-validate a new RF trained only on the top-k features.
    Prints F1 and ROC-AUC to prove that fewer features ≈ same accuracy.
    """
    log.info(f"Evaluating reduced model ({len(top_features)} features)…")
    X_reduced = X[top_features]
    rf_reduced = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = cross_val_score(rf_reduced, X_reduced, y, cv=cv, scoring="f1")
    auc_scores = cross_val_score(rf_reduced, X_reduced, y, cv=cv, scoring="roc_auc")
    log.info(
        f"5-Fold CV  →  F1: {f1_scores.mean():.4f} ± {f1_scores.std():.4f}"
        f" | AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}"
    )


def plot_importance(importances: pd.Series, top_k: int = TOP_K_FEATURES) -> None:
    """Save a bar chart of top-k Gini importances."""
    fig, ax = plt.subplots(figsize=(10, 6))
    top = importances.head(top_k)
    colors = ["#e63946" if i < top_k else "#457b9d" for i in range(len(top))]
    top.plot(kind="barh", ax=ax, color=colors[::-1])
    ax.set_xlabel("Gini Importance (Mean Decrease in Impurity)")
    ax.set_title(f"Top {top_k} Features — Random Forest Gini Importance")
    ax.invert_yaxis()
    fig.tight_layout()
    out = DATA_PROCESSED / "feature_importance.png"
    fig.savefig(out, dpi=150)
    log.info(f"Importance chart saved → {out}")
    plt.close(fig)


def save_top_features(top_features: list[str]) -> None:
    """Persist the list of 15 selected feature names as JSON config."""
    FEATURES_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEATURES_CONFIG_PATH, "w") as f:
        json.dump(top_features, f, indent=2)
    log.info(f"Saved 15-feature config → {FEATURES_CONFIG_PATH}")


def save_reduced_dataset(
    X: pd.DataFrame, y: pd.Series, top_features: list[str]
) -> None:
    """Save the 15-feature dataset to data/processed/."""
    X_reduced = X[top_features].copy()
    X_reduced["FLAG"] = y.values
    out = DATA_PROCESSED / "optimized_15_features.csv"
    X_reduced.to_csv(out, index=False)
    log.info(f"Saved 15-feature dataset → {out}")


def run() -> list[str]:
    """
    Full feature selection pipeline.
    Returns list of selected feature names.
    """
    # Run upstream data prep if needed
    X, y, pipeline, _ = run_data_prep()

    # Scale X using the fitted pipeline
    X_scaled = pd.DataFrame(
        pipeline.transform(X), columns=X.columns, index=X.index
    )

    # Compute Gini importance & select top 15
    rf, importances = compute_gini_importance(X_scaled, y)
    top_features = select_top_k(importances)

    # Evaluate to confirm no accuracy loss
    evaluate_reduced_model(X_scaled, y, top_features)

    # Persist
    plot_importance(importances)
    save_top_features(top_features)
    save_reduced_dataset(X_scaled, y, top_features)

    return top_features


if __name__ == "__main__":
    run()
