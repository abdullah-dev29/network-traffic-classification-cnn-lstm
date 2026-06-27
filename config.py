"""
Single source of truth for all paths, hyperparameters, and toggles.
All other modules import from here — nothing is hardcoded elsewhere.
"""
import os

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
DATA_PATH = "CIC-darknet2020-dataset/cicdarknet2020.parquet"

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

MODEL_PATH      = os.path.join(RESULTS_DIR, "cnn_lstm_model.keras")
SCALER_PATH     = os.path.join(RESULTS_DIR, "scaler.joblib")
HISTORY_PATH    = os.path.join(RESULTS_DIR, "history.json")
TEST_DATA_PATH  = os.path.join(RESULTS_DIR, "test_data.npz")
METRICS_TXT     = os.path.join(RESULTS_DIR, "metrics.txt")
METRICS_JSON    = os.path.join(RESULTS_DIR, "metrics.json")
CONFUSION_PNG   = os.path.join(RESULTS_DIR, "confusion_matrix.png")
CLF_REPORT_TXT  = os.path.join(RESULTS_DIR, "classification_report.txt")

ACC_CURVE_PNG   = os.path.join(FIGURES_DIR, "accuracy_curve.png")
LOSS_CURVE_PNG  = os.path.join(FIGURES_DIR, "loss_curve.png")
ROC_CURVE_PNG   = os.path.join(FIGURES_DIR, "roc_curve.png")

# ---------------------------------------------------------------------------
# Label config
# ---------------------------------------------------------------------------
LABEL_COL     = "Label"
APP_LABEL_COL = "Label.1"  # 8-class application label; needs casing normalisation for Phase 2/3

# Exact strings from the Kaggle cicdarknet2020 build — case- and hyphen-sensitive
DARKNET_CLASSES = ["Tor", "VPN"]
BENIGN_CLASSES  = ["Non-Tor", "NonVPN"]
CLASS_NAMES     = ["Benign", "Darknet"]   # index 0 = Benign, index 1 = Darknet

# ---------------------------------------------------------------------------
# Split / reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.20   # held-out test fraction of the whole dataset
VAL_SIZE     = 0.20   # validation fraction taken from the remaining train portion
# Net split: ~64% train / ~16% val / 20% test

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE = 64   # from the paper
EPOCHS     = 50   # paper's proposed-model ran ~28 epochs; EarlyStopping will cut earlier if needed

# Set to True to pass sklearn class_weight to model.fit — helps if Darknet recall is poor
USE_CLASS_WEIGHT = False

EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE      = 3

# ---------------------------------------------------------------------------
# Architecture (faithful to Figure 7 of Mandela et al. 2025)
# ---------------------------------------------------------------------------
CONV1_FILTERS = 128
CONV1_KERNEL  = 3
CONV2_FILTERS = 64
CONV2_KERNEL  = 3
POOL_SIZE     = 2
LSTM1_UNITS   = 100
LSTM2_UNITS   = 50
DROPOUT_RATE  = 0.3
# "same" keeps temporal length stable for the LSTMs and is robust to ~62 feature count
CONV_PADDING  = "same"

# ---------------------------------------------------------------------------
# Preprocessing toggles
# ---------------------------------------------------------------------------
DROP_CONSTANT_COLS = True   # drop columns with nunique() <= 1 (expect ~15 dropped)