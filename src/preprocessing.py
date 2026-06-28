"""
Data loading and preprocessing for CIC-Darknet2020 binary classification
(Darknet vs Benign) — corrected label-alignment version.

Entry point: load_and_prepare(cfg) -> dict   (same interface as before, so
train.py / evaluate.py / model.py need no changes).

FIX vs the previous version
---------------------------
The old pipeline re-aligned labels to the surviving rows BY POSITION after
dropping duplicates:

    X = X.drop_duplicates()
    y = y[:len(X)]          # WRONG — first len(X) labels by position

drop_duplicates() removes rows from throughout the table, so the survivors
are NOT the first len(X) rows. Slicing y by position silently misaligned
labels with their feature rows and shifted the class counts — that is why
the held-out test set previously held 3,000 Darknet flows instead of the
correct 3,531.

This version keeps y as a pandas Series and re-aligns it BY ROW IDENTITY
after every row drop:

    X = X.loc[~dup_mask]
    y = y.loc[X.index]     # CORRECT — each surviving row keeps its own label

That reproduces the exact same held-out test set as the enhanced
(Transformer) repo — the precondition for a fair baseline-vs-enhanced
comparison.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare(cfg) -> dict:
    # 1. Load
    print(f"Loading dataset from: {cfg.DATA_PATH}")
    df = pd.read_parquet(cfg.DATA_PATH)
    print(f"Loaded shape: {df.shape}  ({df.shape[0]:,} rows x {df.shape[1]} columns)")

    # 2. Build binary target as a pandas Series (index == df.index).
    #    Keeping y as a Series — not a numpy array — is what lets us re-align
    #    it by row identity after dropping rows (step 4).
    darknet_mask = df[cfg.LABEL_COL].isin(cfg.DARKNET_CLASSES)
    benign_mask  = df[cfg.LABEL_COL].isin(cfg.BENIGN_CLASSES)
    unmapped = ~(darknet_mask | benign_mask)
    if unmapped.any():
        bad = df.loc[unmapped, cfg.LABEL_COL].unique().tolist()
        raise AssertionError(
            f"Found {int(unmapped.sum())} rows with unmapped Label values: {bad}. "
            "Update DARKNET_CLASSES / BENIGN_CLASSES in config.py."
        )
    y = darknet_mask.astype(int)  # 1 = Darknet (Tor/VPN), 0 = Benign (Non-Tor/NonVPN)
    print(f"Binary target: Darknet (1) = {int(y.sum()):,}  |  Benign (0) = {int((y == 0).sum()):,}")

    # 3. Drop label columns from the feature matrix
    drop_label_cols = [cfg.LABEL_COL]
    if cfg.APP_LABEL_COL in df.columns:
        drop_label_cols.append(cfg.APP_LABEL_COL)
    X = df.drop(columns=drop_label_cols)
    print(f"Feature shape after dropping label columns: {X.shape}")

    # 4. Defensive cleaning — re-aligning y BY ROW IDENTITY after each drop.
    #    (inf/NaN rows expected = 0; duplicate rows expected ~ 3,099.)
    n_before = len(X)

    X = X.replace([np.inf, -np.inf], np.nan)
    nan_mask = X.isna().any(axis=1)
    X = X.loc[~nan_mask]
    y = y.loc[X.index]                       # re-align
    print(f"Dropped {int(nan_mask.sum())} rows containing inf/NaN")

    dup_mask = X.duplicated()
    X = X.loc[~dup_mask]
    y = y.loc[X.index]                       # re-align  <-- the corrected step
    print(f"Dropped {int(dup_mask.sum())} duplicate rows")
    print(f"Rows: {n_before:,} -> {len(X):,}")

    # 5. Drop constant (zero-variance) columns
    dropped_cols = []
    if cfg.DROP_CONSTANT_COLS:
        dropped_cols = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=dropped_cols)
        print(f"Dropped {len(dropped_cols)} constant column(s): {dropped_cols}")

    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    print(f"Usable feature columns: {n_features}")

    # 6. Stratified split: 64/16/20 train/val/test (same seed as enhanced)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=cfg.VAL_SIZE, stratify=y_trainval,
        random_state=cfg.RANDOM_STATE,
    )

    def _dist(name, s):
        d1 = int(s.sum()); d0 = len(s) - d1
        print(f"  {name:8s}: {len(s):7,} rows  |  Darknet={d1:,} ({100*d1/len(s):.1f}%)  "
              f"Benign={d0:,} ({100*d0/len(s):.1f}%)")
    print("Split class distributions (verify test Darknet = 3,531):")
    _dist("train", y_train); _dist("val", y_val); _dist("test", y_test)

    # 7. Scale — fit on TRAIN ONLY to prevent leakage
    scaler_dir = os.path.dirname(cfg.SCALER_PATH)
    if scaler_dir:
        os.makedirs(scaler_dir, exist_ok=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, cfg.SCALER_PATH)
    print(f"Scaler fitted on train only and saved to: {cfg.SCALER_PATH}")

    # 8. Reshape to 3D for Conv1D: (n_samples, n_features, 1)
    X_train_3d = X_train_scaled.reshape(-1, n_features, 1)
    X_val_3d   = X_val_scaled.reshape(-1, n_features, 1)
    X_test_3d  = X_test_scaled.reshape(-1, n_features, 1)
    print(f"Final shapes — X_train: {X_train_3d.shape}  X_val: {X_val_3d.shape}  X_test: {X_test_3d.shape}")

    return {
        "X_train": X_train_3d, "y_train": y_train.to_numpy(),
        "X_val":   X_val_3d,   "y_val":   y_val.to_numpy(),
        "X_test":  X_test_3d,  "y_test":  y_test.to_numpy(),
        "n_features": n_features,
        "feature_names": feature_names,
        "dropped_cols": dropped_cols,
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config as cfg
    out = load_and_prepare(cfg)
    print(f"\nn_features={out['n_features']}, dropped={len(out['dropped_cols'])} constant cols")