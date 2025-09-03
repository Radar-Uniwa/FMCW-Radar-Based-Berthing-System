import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
BASE           = r"path/to/folder"  # adjust to your root folder
SETS           = ["Training", "Validation", "Test"] # Adjust as needed based on the names
FEATURE_SUBDIR = "features"
SCALER_OUT     = os.path.join(BASE, "name_of_scaler.pkl")# abjust based on the name of scaler
MODEL_OUT      = os.path.join(BASE, "name_of_model.pkl")# abjust based on the name of scaler

# ─── 1) LOAD & CONCAT ───────────────────────────────────────────────────────────
data = {}
for split in SETS:
    pattern = os.path.join(BASE, split, FEATURE_SUBDIR, "*_features.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No feature CSVs found in {pattern}")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    data[split] = df
    print(f"{split}: {len(df)} rows loaded")

# ─── 2) SPLIT X / y ──────────────────────────────────────────────────────────────
X, y = {}, {}
for split, df in data.items():
    y[split] = df["Dock_Label"].values
    X[split] = df.drop(columns=["Frame_ID", "Tracking_ID", "Dock_Label", "X", "Y", "Z"])

# ─── 3) SCALE ───────────────────────────────────────────────────────────────────
scaler = StandardScaler().fit(X["Training"])
joblib.dump(scaler, SCALER_OUT)
print(f"✅ Scaler saved to {SCALER_OUT}")

X_train = scaler.transform(X["Training_2"]) # Adjust as needed based on the line 21
X_val   = scaler.transform(X["Validation_2"]) # Adjust as needed based on the line 21
X_test  = scaler.transform(X["Test_2"]) # Adjust as needed based on the line 21

# ─── 4) TRAIN BALANCED RANDOM FOREST ─────────────────────────────────────────────
clf = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y["Training"])
joblib.dump(clf, MODEL_OUT)
print(f"✅ Model saved to {MODEL_OUT}")

# ─── 5) PREDICTIONS ─────────────────────────────────────────────────────────────
val_pred  = clf.predict(X_val)
test_pred = clf.predict(X_test)

# ─── 6) PRINT BASIC METRICS ─────────────────────────────────────────────────────
print("\n=== Validation Metrics ===")
print("Accuracy:", accuracy_score(y["Validation"], val_pred))
print(classification_report(y["Validation"], val_pred,
                            target_names=["No Dock", "Dock"]))

print("\n=== Test Metrics ===")
print("Accuracy:", accuracy_score(y["Test_2"], test_pred))
print(classification_report(y["Test_2"], test_pred,
                            target_names=["No Dock", "Dock"]))

# ─── 7) CONFUSION MATRICES AS HEATMAPS WITH DYNAMIC TEXT COLORS ───────────────
for split, true, pred in [
    ("Validation", y["Validation"], val_pred),
    ("Test",       y["Test"],       test_pred)
]:
    cm = confusion_matrix(true, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Dock", "Dock"],
        yticklabels=["No Dock", "Dock"],
        ax=ax
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",     fontsize=12)
    ax.set_title(f"{split} Confusion Matrix", fontsize=14)

    # White text on dark cells, black on light
    thresh = cm.max() / 2
    for txt in ax.texts:
        val = int(txt.get_text())
        color = "white" if val > thresh else "black"
        txt.set_color(color)
        txt.set_fontsize(14)
        txt.set_fontweight("bold")

    plt.tight_layout()
    plt.show()

# ─── 8) PRECISION / RECALL / F1 METRICS MATRIX ──────────────────────────────────
report_dict = classification_report(
    y["Test"], test_pred,
    target_names=["No Dock", "Dock"],
    output_dict=True
)
metrics_df = pd.DataFrame(report_dict).T.drop(columns=["support"])
metrics_df = metrics_df.loc[["No Dock", "Dock"], ["precision", "recall", "f1-score"]]
metrics_df = metrics_df.round(2)

fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(
    metrics_df,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar=False,
    annot_kws={"size":12, "weight":"bold"},
    linewidths=0.5,
    ax=ax
)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
ax.set_title("Test Set: Precision / Recall / F1-Score", fontsize=14)
plt.tight_layout()
plt.show()

# ─── FEATURE IMPORTANCES (HORIZONTAL BAR) ───────────────────────────────────
feature_names = X["Training"].columns.tolist()
importances   = clf.feature_importances_
feat_series   = pd.Series(importances, index=feature_names)

# Select top 20
topk = feat_series.nlargest(20).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(topk.index, topk.values, color=sns.color_palette("Blues", n_colors=20))

# Increase tick label size
ax.tick_params(axis="y", labelsize=13)
ax.tick_params(axis="x", labelsize=12)

# Title & axis labels a bit bigger
ax.set_title("Top 20 Feature Importances (Random Forest)", fontsize=16, pad=15)
ax.set_xlabel("Importance", fontsize=14)
ax.set_ylabel("")

# Annotate each bar *inside* with white text if bar is wide enough
for bar in bars:
    w = bar.get_width()
    y = bar.get_y() + bar.get_height()/2
    txt = f"{w:.2%}"
    # choose white if the bar is wide, else black
    color = "white" if w > topk.max() * 0.15 else "black"
    ax.text(
        w * 0.98,   # 98% of the bar width → inside right
        y,
        txt,
        va="center",
        ha="right",
        fontsize=13,
        fontweight="bold",
        color=color
    )

plt.tight_layout()
plt.show()
