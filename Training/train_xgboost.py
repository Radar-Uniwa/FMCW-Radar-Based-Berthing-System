# tune_xgboost.py

import os
import glob
import joblib
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ─── CONFIGURATION ──────────────────────────────────────────────────────────────
BASE_DIR       = r"path/to/folder"   # adjust to your root folder
SETS           = ["Training_2", "Validation_2", "Test_2"] # Adjust as needed based on the names
FEATURE_SUBDIR = "features"                    # subfolder containing *_features.csv
SCALER_OUT     = os.path.join(BASE_DIR, "name_of_scaler.pkl") # abjust based on the name of scaler
MODEL_OUT      = os.path.join(BASE_DIR, "name_of_model.pkl")# abjust based on the name of scaler

# ─── 1) LOAD FEATURES ────────────────────────────────────────────────────────────
def load_feature_split(split):
    pattern = os.path.join(BASE_DIR, split, FEATURE_SUBDIR, "*_features.csv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No feature CSVs found in {pattern}")
    df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in files],
                   ignore_index=True)
    y = df["Dock_Label"].values
    Xdf = df.drop(columns=["Frame_ID","Tracking_ID","Dock_Label","X","Y","Z"])
    X  = Xdf.values
    print(f"{split}: loaded {len(df)} rows")
    return X, y, Xdf.columns.tolist()

X_train, y_train, feature_names = load_feature_split("Training_2") # Adjust as needed based on the line 28
X_val,   y_val,   _             = load_feature_split("Validation_2")# Adjust as needed based on the line 28
X_test,  y_test,  _             = load_feature_split("Test_2")# Adjust as needed based on the line 28

# ─── 2) SCALE ───────────────────────────────────────────────────────────────────
scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, SCALER_OUT)
print(f"✅ Scaler saved to {SCALER_OUT}")
X_train = scaler.transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

# ─── 3) CROSS-VALIDATION & SCORER ───────────────────────────────────────────────
cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
f1_scorer = make_scorer(f1_score, pos_label=1)

# ─── 4) COMPUTE scale_pos_weight ────────────────────────────────────────────────
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.2f}")

# ─── 5) BASE XGB CLASSIFIER ──────────────────────────────────────────────────────
base_xgb = XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

# ─── 6) HYPERPARAMETER SPACE ────────────────────────────────────────────────────
param_dist = {
    "n_estimators":     [100,200,300,500],
    "max_depth":        [4,6,8,10],
    "learning_rate":    [0.01,0.05,0.1,0.2],
    "subsample":        [0.6,0.8,1.0],
    "colsample_bytree": [0.6,0.8,1.0],
    "gamma":            [0,1,5],
}

# ─── 7) RANDOMIZED SEARCH ───────────────────────────────────────────────────────
search = RandomizedSearchCV(
    estimator           = base_xgb,
    param_distributions = param_dist,
    n_iter              = 30,
    scoring             = f1_scorer,
    cv                  = cv,
    verbose             = 2,
    random_state        = 42,
    n_jobs              = -1
)
print("Starting hyperparameter search...")
search.fit(X_train, y_train)
print("\n🏆 Best CV F1-score:", search.best_score_)
print("🔧 Best hyperparameters:", search.best_params_)

# ─── 8) TRAIN FINAL MODEL ───────────────────────────────────────────────────────
best_model = search.best_estimator_
best_model.fit(X_train, y_train)
joblib.dump(best_model, MODEL_OUT)
print(f"✅ Tuned model saved to {MODEL_OUT}")

# ─── 9) EVALUATION FUNCTION ────────────────────────────────────────────────────
def evaluate_split(name, X, y, clf):
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:,1]
    print(f"\n=== {name} Metrics ===")
    print("Accuracy:", accuracy_score(y, preds))
    print(classification_report(y, preds, target_names=["No Dock","Dock"]))

    # ROC & AUC
    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'k--')
    plt.title(f"{name} ROC")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # Precision-Recall & AP
    prec, rec, _ = precision_recall_curve(y, probs)
    ap = average_precision_score(y, probs)
    plt.figure(figsize=(5,4))
    plt.plot(rec, prec, label=f"AP={ap:.3f}")
    plt.title(f"{name} Precision-Recall")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

evaluate_split("Validation", X_val, y_val, best_model)
evaluate_split("Test",       X_test, y_test,   best_model)

# ─── 10) PREDICTIONS ─────────────────────────────────────────────────────────────
val_pred  = best_model.predict(X_val)
test_pred = best_model.predict(X_test)

# ─── A) STYLED CONFUSION MATRICES WITH COUNTS + PERCENTAGES ────────────────────
for split, true, pred in [
    ("Validation", y_val,  val_pred),
    ("Test",       y_test, test_pred)
]:
    cm = confusion_matrix(true, pred)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_perc  = cm / row_sums * 100

    # annotation strings
    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i,j] = f"{cm[i,j]}\n({cm_perc[i,j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Dock","Dock"],
        yticklabels=["No Dock","Dock"],
        annot_kws={"fontsize":12, "weight":"bold"},
        ax=ax
    )
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label",     fontsize=12)
    ax.set_title(f"{split} Confusion Matrix", fontsize=14)

    thresh = cm.max()/2
    for text in ax.texts:
        count = int(text.get_text().split("\n",1)[0])
        text.set_color("white" if count>thresh else "black")

    plt.tight_layout()
    plt.show()

# ─── B) METRICS HEATMAP (Test only, in %) ───────────────────────────────────────
report_dict = classification_report(
    y_test,
    test_pred,
    target_names=["No Dock","Dock"],
    output_dict=True
)
metrics_df = pd.DataFrame(report_dict).T.drop(columns=["support"])
metrics_df = metrics_df.loc[["No Dock","Dock"], ["precision","recall","f1-score"]]
metrics_pct = (metrics_df * 100).round(1)

# build annotation strings with percent sign
annot = np.empty(metrics_pct.shape, dtype=object)
for i, row in enumerate(metrics_pct.index):
    for j, col in enumerate(metrics_pct.columns):
        annot[i,j] = f"{metrics_pct.iloc[i,j]:.1f}%"

fig, ax = plt.subplots(figsize=(6,3))
sns.heatmap(
    metrics_pct,
    annot=annot,
    fmt="",
    cmap="Blues",
    cbar=True,
    cbar_kws={'label':'%', 'shrink':0.8},
    annot_kws={"size":12, "weight":"bold"},
    linewidths=0.5,
    xticklabels=metrics_pct.columns,
    yticklabels=metrics_pct.index,
    ax=ax
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
ax.set_title("Test Set: Precision / Recall / F1-Score", fontsize=14)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.show()

# ─── C) FEATURE IMPORTANCES (HORIZONTAL BAR) ──────────────────────────────────
booster     = best_model.get_booster()
gain_dict   = booster.get_score(importance_type="gain")
gain_series = pd.Series(gain_dict).sort_values(ascending=False)
topk        = gain_series.head(20).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(8,6))
bars = ax.barh(topk.index, topk.values, color=sns.color_palette("Blues", n_colors=20))
ax.set_title("XGBoost Feature Importances (gain)", fontsize=16, pad=15)
ax.set_xlabel("Gain", fontsize=14)
ax.tick_params(axis="y", labelsize=13)
ax.tick_params(axis="x", labelsize=12)

max_gain = topk.max()
for bar in bars:
    w = bar.get_width()
    y = bar.get_y() + bar.get_height()/2
    txt = f"{w:.3f}"
    color = "white" if w > max_gain * 0.15 else "black"
    ax.text(
        w * 0.98,
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
