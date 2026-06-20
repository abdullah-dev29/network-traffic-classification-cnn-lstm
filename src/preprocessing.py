"""
Data loading and preprocessing pipeline for CIC-Darknet2020 binary classification.

Entry point: load_and_prepare(cfg) -> dict
Side-effect-free on import; all work happens inside functions.
"""
import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare(cfg) -> dict:
    """
    Load the parquet, engineer the binary target, clean, split, scale, and reshape.

    Parameters
    ----------
    cfg : module
        The config module (or any object with the required attributes).

    Returns
    -------
    dict with keys:
        X_train, y_train, X_val, y_val, X_test, y_test  — numpy arrays
        n_features     — int, number of features after dropping constants
        feature_names  — list[str], surviving feature column names
        dropped_cols   — list[str], constant columns that were removed
    """
    # ------------------------------------------------------------------
    # 1. Load
    # ------------------------------------------------------------------
    print(f"Loading dataset from: {cfg.DATA_PATH}")
    df = pd.read_parquet(cfg.DATA_PATH)
    print(f"Loaded shape: {df.shape}  ({df.shape[0]:,} rows × {df.shape[1]} columns)")

    # ------------------------------------------------------------------
    # 2. Build binary target y
    # ------------------------------------------------------------------
    label_series = df[cfg.LABEL_COL]
    all_labels = set(label_series.unique())
    known_labels = set(cfg.DARKNET_CLASSES) | set(cfg.BENIGN_CLASSES)

    unmapped = all_labels - known_labels
    assert not unmapped, (
        f"Unmapped label values found: {unmapped}. "
        "Update DARKNET_CLASSES / BENIGN_CLASSES in config.py."
    )

    y = label_series.map(
        {lbl: 1 for lbl in cfg.DARKNET_CLASSES} |
        {lbl: 0 for lbl in cfg.BENIGN_CLASSES}
    ).to_numpy(dtype=np.int32)

    assert not np.isnan(y).any(), "y contains NaN after mapping — mapping failed."
    assert set(y).issubset({0, 1}), "y contains values outside {0, 1}."

    darknet_count = int(y.sum())
    benign_count  = int((y == 0).sum())
    print(f"Binary target: Darknet (1) = {darknet_count:,}  |  Benign (0) = {benign_count:,}")

    # ------------------------------------------------------------------
    # 3. Drop label columns
    # ------------------------------------------------------------------
    X = df.drop(columns=[cfg.LABEL_COL, cfg.APP_LABEL_COL])
    print(f"Feature shape after dropping label columns: {X.shape}")

    # ------------------------------------------------------------------
    # 4. Defensive cleaning (dataset is pre-cleaned; expect 0 changes)
    # ------------------------------------------------------------------
    n_before = len(X)

    # Replace ±inf with NaN then drop those rows
    X = X.replace([np.inf, -np.inf], np.nan)
    n_after_inf = X.dropna().shape[0]
    inf_nan_removed = n_before - n_after_inf
    X = X.dropna()
    y = y[X.index] if isinstance(X.index, pd.Index) else y[:len(X)]

    # Re-align y to X after row drops
    y = y[:len(X)]

    # Drop duplicate rows
    before_dedup = len(X)
    X = X.drop_duplicates()
    dup_removed = before_dedup - len(X)
    y = y[:len(X)]

    print(f"Defensive cleaning: removed {inf_nan_removed} inf/NaN rows, {dup_removed} duplicate rows")

    # ------------------------------------------------------------------
    # 5. Drop constant columns
    # ------------------------------------------------------------------
    dropped_cols: list[str] = []
    if cfg.DROP_CONSTANT_COLS:
        nunique = X.nunique()
        constant_mask = nunique <= 1
        dropped_cols = X.columns[constant_mask].tolist()
        X = X.drop(columns=dropped_cols)
        print(f"Dropped {len(dropped_cols)} constant column(s): {dropped_cols}")
        print(f"Feature shape after dropping constants: {X.shape}  ({X.shape[1]} features remain)")

    feature_names = X.columns.tolist()
    n_features = X.shape[1]

    # ------------------------------------------------------------------
    # 6. Stratified splits
    # ------------------------------------------------------------------
    X_arr = X.to_numpy(dtype=np.float32)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_arr, y,
        test_size=cfg.TEST_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=cfg.VAL_SIZE,
        stratify=y_trainval,
        random_state=cfg.RANDOM_STATE,
    )

    def _class_dist(arr, name):
        d1 = arr.sum(); d0 = len(arr) - d1
        print(f"  {name:8s}: {len(arr):7,} rows  |  Darknet={d1:,} ({100*d1/len(arr):.1f}%)  Benign={d0:,} ({100*d0/len(arr):.1f}%)")

    print("Split class distributions (stratification check):")
    _class_dist(y_train, "train")
    _class_dist(y_val,   "val")
    _class_dist(y_test,  "test")

    # ------------------------------------------------------------------
    # 7. Scale — fit on TRAIN ONLY to prevent data leakage
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(cfg.SCALER_PATH) if os.path.dirname(cfg.SCALER_PATH) else ".", exist_ok=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    joblib.dump(scaler, cfg.SCALER_PATH)
    print(f"Scaler fitted on train set and saved to: {cfg.SCALER_PATH}")

    # ------------------------------------------------------------------
    # 8. Reshape to 3D for Conv1D: (n_samples, n_features, 1)
    # ------------------------------------------------------------------
    X_train = X_train.reshape(-1, n_features, 1)
    X_val   = X_val.reshape(-1, n_features, 1)
    X_test  = X_test.reshape(-1, n_features, 1)

    print(f"Final shapes — X_train: {X_train.shape}  X_val: {X_val.shape}  X_test: {X_test.shape}")

    return {
        "X_train":      X_train,
        "y_train":      y_train,
        "X_val":        X_val,
        "y_val":        y_val,
        "X_test":       X_test,
        "y_test":       y_test,
        "n_features":   n_features,
        "feature_names": feature_names,
        "dropped_cols": dropped_cols,
    }


if __name__ == "__main__":
    # Cheap demo: load data with pandas/sklearn only — no Keras, no training.
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import config as cfg
    result = load_and_prepare(cfg)
    print(f"\nn_features={result['n_features']}, dropped={len(result['dropped_cols'])} constant cols")
    print("Dropped columns:", result["dropped_cols"])
