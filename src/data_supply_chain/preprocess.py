from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from data_supply_chain._knn import (
        gpu_backend_available,
        last_backend,
        soft_knn_impute_cpu,
        soft_knn_impute_from_reference_cpu,
        soft_knn_impute_from_reference_gpu,
        soft_knn_impute_gpu,
        time_imputation,
    )
    HAS_SOFT_KNN = True
except ImportError:
    HAS_SOFT_KNN = False

LAB_COLS = ["acr", "scys", "diabetes", "hypertension"]


def distribution_variant() -> str:
    return "full" if HAS_SOFT_KNN else "workshop"


def add_missingness_indicators(
    df: pd.DataFrame, cols: list[str] | None = None
) -> pd.DataFrame:
    cols = cols or LAB_COLS
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)
    return out


def build_feature_matrix(
    df: pd.DataFrame,
    egfr_col: str,
    include_labs: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    base = ["age", "female", egfr_col]
    labs = []
    if include_labs:
        for col in LAB_COLS:
            if col in df.columns:
                labs.append(col)
                labs.append(f"{col}_missing")
    cols = base + labs
    return df[cols].astype(float), cols


def median_impute(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    col_medians = np.nanmedian(X, axis=0)
    return np.where(np.isnan(X), col_medians, X)


def impute_features(
    X: np.ndarray,
    k: int = 5,
    tau: float = 0.5,
    use_gpu: bool = False,
) -> np.ndarray:
    if not HAS_SOFT_KNN:
        out = median_impute(X)
    elif use_gpu:
        out = soft_knn_impute_gpu(X, k=k, tau=tau)
    else:
        out = soft_knn_impute_cpu(X, k=k, tau=tau)
    if np.isnan(out).any():
        col_medians = np.nanmedian(np.where(np.isnan(X), np.nanmedian(X, axis=0), X), axis=0)
        for j in range(out.shape[1]):
            miss = np.isnan(out[:, j])
            if miss.any():
                out[miss, j] = col_medians[j]
    return out


def median_impute_from_reference(X_query: np.ndarray, X_ref: np.ndarray) -> np.ndarray:
    X_query = np.asarray(X_query, dtype=float)
    col_medians = np.nanmedian(np.asarray(X_ref, dtype=float), axis=0)
    return np.where(np.isnan(X_query), col_medians, X_query)


def impute_query_from_reference(
    X_query: np.ndarray,
    X_ref: np.ndarray,
    k: int = 5,
    tau: float = 0.5,
    use_gpu: bool = False,
) -> np.ndarray:
    if not HAS_SOFT_KNN:
        out = median_impute_from_reference(X_query, X_ref)
    elif use_gpu:
        out = soft_knn_impute_from_reference_gpu(X_query, X_ref, k=k, tau=tau)
    else:
        out = soft_knn_impute_from_reference_cpu(X_query, X_ref, k=k, tau=tau)
    if np.isnan(out).any():
        col_medians = np.nanmedian(np.asarray(X_ref, dtype=float), axis=0)
        for j in range(out.shape[1]):
            miss = np.isnan(out[:, j])
            if miss.any() and np.isfinite(col_medians[j]):
                out[miss, j] = col_medians[j]
    return out


def impute_train_test(
    X: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    k: int = 5,
    tau: float = 0.5,
    use_gpu: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)
    X_train_imp = impute_features(X[train_idx], k=k, tau=tau, use_gpu=use_gpu)
    X_test_imp = impute_query_from_reference(
        X[test_idx], X[train_idx], k=k, tau=tau, use_gpu=use_gpu
    )
    return X_train_imp, X_test_imp
