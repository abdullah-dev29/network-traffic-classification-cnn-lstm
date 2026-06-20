"""
Training script — runs on Google Colab (GPU), NOT on this laptop.

Usage (from repo root on Colab):
    python src/train.py

Saves to results/:
    cnn_lstm_model.keras  — trained model
    scaler.joblib         — fitted StandardScaler (already saved by preprocessing)
    history.json          — per-epoch metrics
    test_data.npz         — scaled + reshaped X_test, y_test for evaluate.py
"""
import os
import sys
import json
import random
import numpy as np

# Ensure repo root is on the path so `config` and `src.*` are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

# ------------------------------------------------------------------
# Seed everything for reproducibility.
# Note: full GPU determinism is not guaranteed even with fixed seeds.
# ------------------------------------------------------------------
random.seed(cfg.RANDOM_STATE)
np.random.seed(cfg.RANDOM_STATE)

import tensorflow as tf
try:
    tf.keras.utils.set_random_seed(cfg.RANDOM_STATE)
except AttributeError:
    tf.random.set_seed(cfg.RANDOM_STATE)

# ------------------------------------------------------------------
# Create output directories
# ------------------------------------------------------------------
os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
os.makedirs(cfg.FIGURES_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------
from src.preprocessing import load_and_prepare

print("=" * 60)
print("STEP 1: Preprocessing")
print("=" * 60)
data = load_and_prepare(cfg)

X_train, y_train = data["X_train"], data["y_train"]
X_val,   y_val   = data["X_val"],   data["y_val"]
X_test,  y_test  = data["X_test"],  data["y_test"]
n_features       = data["n_features"]

# ------------------------------------------------------------------
# Build model
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 2: Building model")
print("=" * 60)
from src.model import build_cnn_lstm
model = build_cnn_lstm(n_features=n_features)

# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=cfg.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        patience=cfg.REDUCE_LR_PATIENCE,
        factor=0.5,
        verbose=1,
    ),
]

# ------------------------------------------------------------------
# Optional class weights (default OFF to match the paper)
# ------------------------------------------------------------------
class_weight = None
if cfg.USE_CLASS_WEIGHT:
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    class_weight = {0: weights[0], 1: weights[1]}
    print(f"Using class weights: {class_weight}")

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 3: Training")
print("=" * 60)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=cfg.EPOCHS,
    batch_size=cfg.BATCH_SIZE,
    callbacks=callbacks,
    class_weight=class_weight,
    verbose=1,
)

# ------------------------------------------------------------------
# Persist artifacts for evaluate.py
# ------------------------------------------------------------------
print("\n" + "=" * 60)
print("STEP 4: Saving artifacts")
print("=" * 60)

model.save(cfg.MODEL_PATH)
print(f"Model saved → {cfg.MODEL_PATH}")

with open(cfg.HISTORY_PATH, "w") as f:
    json.dump(history.history, f, indent=2)
print(f"History saved → {cfg.HISTORY_PATH}")

np.savez_compressed(cfg.TEST_DATA_PATH, X_test=X_test, y_test=y_test)
print(f"Test data saved → {cfg.TEST_DATA_PATH}")

# ------------------------------------------------------------------
# Final-epoch summary (evaluate.py does the full reporting)
# ------------------------------------------------------------------
last = {k: v[-1] for k, v in history.history.items()}
print("\nFinal epoch metrics:")
for k, v in last.items():
    print(f"  {k}: {v:.4f}")
print("\nTraining complete. Run src/evaluate.py for full metrics and figures.")
