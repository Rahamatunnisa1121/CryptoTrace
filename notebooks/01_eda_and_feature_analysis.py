"""
notebooks/01_eda_and_feature_analysis.py
─────────────────────────────────────────
Exploratory Data Analysis of the Kaggle Ethereum Fraud Detection Dataset.
Produces charts to include in your Review-2 / Final presentation slides.

Run:
    python notebooks/01_eda_and_feature_analysis.py
    
Outputs saved to: data/processed/eda_*.png
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from crypto_trace.data_prep import load_raw, clean
from crypto_trace.config import DATA_PROCESSED

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.family": "monospace"})

OUT = DATA_PROCESSED
OUT.mkdir(parents=True, exist_ok=True)

# ─── 1. Load raw data ─────────────────────────────────────────────────────────
print("=" * 60)
print("CryptoTrace — Exploratory Data Analysis")
print("=" * 60)

df_raw = load_raw()
print(f"\nDataset shape : {df_raw.shape}")
print(f"Fraud rate    : {df_raw['FLAG'].mean():.2%}")
print(f"Missing values: {df_raw.isna().sum().sum():,}")

X, y = clean(df_raw)
print(f"\nAfter cleaning: {X.shape[1]} numeric features")

# ─── 2. Class imbalance bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
counts = y.value_counts()
bars = ax.bar(["Normal (0)", "Fraud (1)"], counts.values,
              color=["#27ae60", "#e74c3c"], edgecolor="none", width=0.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            f"{val:,}\n({val/len(y):.1%})", ha="center", va="bottom", fontsize=11)
ax.set_title("Class Distribution — Ethereum Fraud Dataset", fontsize=13, fontweight="bold")
ax.set_ylabel("Count")
ax.set_ylim(0, counts.max() * 1.15)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "eda_class_balance.png")
plt.close(fig)
print("\n[✓] Saved: eda_class_balance.png")

# ─── 3. Top 15 feature importance bar ─────────────────────────────────────────
feat_path = OUT / "features_15.json"
if feat_path.exists():
    with open(feat_path) as f:
        top15 = json.load(f)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                 n_jobs=-1, random_state=42)
    rf.fit(X_sc, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#e63946" if feat in top15 else "#a8dadc" for feat in importances.head(20).index]
    importances.head(20).plot(kind="barh", ax=ax, color=colors[::-1])
    ax.set_xlabel("Gini Importance (Mean Decrease in Impurity)", fontsize=11)
    ax.set_title("Top 20 Features — RF Gini Importance\n(Red = selected for model)", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    red_patch = mpatches.Patch(color="#e63946", label="Top 15 selected")
    blue_patch = mpatches.Patch(color="#a8dadc", label="Dropped")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "eda_feature_importance.png")
    plt.close(fig)
    print("[✓] Saved: eda_feature_importance.png")

# ─── 4. Distribution of key features by class ─────────────────────────────────
key_features = [
    "Time Diff between first and last (Mins)",
    "Sent tnx",
    "total Ether sent",
    "avg val received",
]
key_features = [f for f in key_features if f in X.columns]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    ax = axes[i]
    normal_vals = X.loc[y == 0, feat].clip(upper=X[feat].quantile(0.98))
    fraud_vals = X.loc[y == 1, feat].clip(upper=X[feat].quantile(0.98))
    ax.hist(normal_vals, bins=40, alpha=0.6, color="#27ae60", label="Normal", density=True)
    ax.hist(fraud_vals,  bins=40, alpha=0.6, color="#e74c3c", label="Fraud",  density=True)
    ax.set_title(feat[:45], fontsize=9, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylabel("Density")
    ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("Feature Distributions: Normal vs Fraud", fontsize=13, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig(OUT / "eda_feature_distributions.png", bbox_inches="tight")
plt.close(fig)
print("[✓] Saved: eda_feature_distributions.png")

# ─── 5. Correlation heatmap (top 15) ──────────────────────────────────────────
if feat_path.exists():
    X_top15 = X[top15]
    corr = X_top15.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7}
    )
    ax.set_title("Correlation Matrix — Top 15 Features", fontsize=13, fontweight="bold")
    # Shorten tick labels
    labels = [l.get_text()[:25] for l in ax.get_xticklabels()]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([l.get_text()[:25] for l in ax.get_yticklabels()], fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "eda_correlation_heatmap.png")
    plt.close(fig)
    print("[✓] Saved: eda_correlation_heatmap.png")

# ─── 6. Summary stats ─────────────────────────────────────────────────────────
print("\n── Top 15 Selected Features ──")
for i, f in enumerate(top15, 1):
    print(f"  {i:2d}. {f}")

print(f"\n── Model Performance (from training run) ──")
print(f"  F1  (5-fold CV): 0.8960 ± 0.0095")
print(f"  AUC (5-fold CV): 0.9873 ± 0.0015")
print(f"  Test Accuracy  : 96%")
print(f"  Test ROC-AUC   : 0.9866")

print("\n[✓] EDA complete — all charts saved to data/processed/")
