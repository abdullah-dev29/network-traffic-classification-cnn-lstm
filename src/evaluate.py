"""
Evaluation script — runs on Google Colab after train.py completes.

Usage (from repo root on Colab):
    python src/evaluate.py

Loads artifacts from results/ (written by train.py) and produces:
    results/metrics.txt              — Table-2-style summary (6 metrics)
    results/metrics.json             — same metrics as JSON
    results/classification_report.txt
    results/confusion_matrix.png
    figures/accuracy_curve.png
    figures/loss_curve.png
    figures/roc_curve.png
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg

import tensorflow as tf
import matplotlib
matplotlib.use("Agg")  # headless — no display needed on Colab
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. Load artifacts
# ------------------------------------------------------------------
print("Loading model …")
model = tf.keras.models.load_model(cfg.MODEL_PATH)

print("Loading test data …")
npz = np.load(cfg.TEST_DATA_PATH)
X_test, y_test = npz["X_test"], npz["y_test"]

print("Loading training history …")
with open(cfg.HISTORY_PATH) as f:
    history = json.load(f)

# ------------------------------------------------------------------
# 2. Predict
# ------------------------------------------------------------------
y_prob = model.predict(X_test, batch_size=cfg.BATCH_SIZE, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)

# ------------------------------------------------------------------
# 3. Compute the six metrics reported in the paper's Table 2
# ------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy    = accuracy_score(y_test, y_pred)
precision   = precision_score(y_test, y_pred, zero_division=0)
recall      = recall_score(y_test, y_pred, zero_division=0)      # sensitivity
f1          = f1_score(y_test, y_pred, zero_division=0)
auc         = roc_auc_score(y_test, y_prob)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0           # TN / (TN + FP)

metrics = {
    "accuracy":    round(float(accuracy),    4),
    "f1_score":    round(float(f1),          4),
    "recall":      round(float(recall),      4),
    "precision":   round(float(precision),   4),
    "roc_auc":     round(float(auc),         4),
    "specificity": round(float(specificity), 4),
}

# ------------------------------------------------------------------
# 4a. Write metrics.txt — mirrors the paper's Table 2 layout
# ------------------------------------------------------------------
header = f"{'Model':<35} {'Accuracy':>10} {'F1':>8} {'Recall':>8} {'Precision':>10} {'AUC':>8} {'Specificity':>12}"
row    = (
    f"{'Proposed CNN-LSTM (rebuild)':<35} "
    f"{metrics['accuracy']:>10.4f} "
    f"{metrics['f1_score']:>8.4f} "
    f"{metrics['recall']:>8.4f} "
    f"{metrics['precision']:>10.4f} "
    f"{metrics['roc_auc']:>8.4f} "
    f"{metrics['specificity']:>12.4f}"
)
sep = "-" * len(header)
txt_content = f"{header}\n{sep}\n{row}\n"

with open(cfg.METRICS_TXT, "w") as f:
    f.write(txt_content)
print(f"Metrics table saved → {cfg.METRICS_TXT}")

# ------------------------------------------------------------------
# 4b. Write metrics.json
# ------------------------------------------------------------------
with open(cfg.METRICS_JSON, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics JSON saved → {cfg.METRICS_JSON}")

# ------------------------------------------------------------------
# 4c. Write classification_report.txt
# ------------------------------------------------------------------
report = classification_report(y_test, y_pred, target_names=cfg.CLASS_NAMES, zero_division=0)
with open(cfg.CLF_REPORT_TXT, "w") as f:
    f.write(report)
print(f"Classification report saved → {cfg.CLF_REPORT_TXT}")

# ------------------------------------------------------------------
# 5. Confusion matrix figure
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=cfg.CLASS_NAMES, yticklabels=cfg.CLASS_NAMES,
    ax=ax,
)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Confusion Matrix — CNN-LSTM (CIC-Darknet2020)")
fig.tight_layout()
fig.savefig(cfg.CONFUSION_PNG, dpi=150)
plt.close(fig)
print(f"Confusion matrix saved → {cfg.CONFUSION_PNG}")

# ------------------------------------------------------------------
# 6. ROC curve
# ------------------------------------------------------------------
from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr, tpr, label=f"CNN-LSTM (AUC = {auc:.4f})", linewidth=2)
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — CNN-LSTM")
ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(cfg.ROC_CURVE_PNG, dpi=150)
plt.close(fig)
print(f"ROC curve saved → {cfg.ROC_CURVE_PNG}")

# ------------------------------------------------------------------
# 7. Accuracy curve
# ------------------------------------------------------------------
acc     = history.get("accuracy", [])
val_acc = history.get("val_accuracy", [])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(acc,     label="Train accuracy")
ax.plot(val_acc, label="Val accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_title("Training vs Validation Accuracy")
ax.legend()
fig.tight_layout()
fig.savefig(cfg.ACC_CURVE_PNG, dpi=150)
plt.close(fig)
print(f"Accuracy curve saved → {cfg.ACC_CURVE_PNG}")

# ------------------------------------------------------------------
# 8. Loss curve
# ------------------------------------------------------------------
loss     = history.get("loss", [])
val_loss = history.get("val_loss", [])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(loss,     label="Train loss")
ax.plot(val_loss, label="Val loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Training vs Validation Loss")
ax.legend()
fig.tight_layout()
fig.savefig(cfg.LOSS_CURVE_PNG, dpi=150)
plt.close(fig)
print(f"Loss curve saved → {cfg.LOSS_CURVE_PNG}")

# ------------------------------------------------------------------
# 9. Human-readable summary to stdout
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(txt_content)
print("Classification report:")
print(report)
