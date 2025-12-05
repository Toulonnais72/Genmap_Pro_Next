from __future__ import annotations

from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

__all__ = [
    "compute_knn_applicability_domain",
    "compute_ad_for_dataframe",
]


def compute_knn_applicability_domain(
    X_train: np.ndarray,
    X_new: np.ndarray,
    n_neighbors: int = 5,
    quantiles: Tuple[float, float] = (0.80, 0.95),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute applicability domain (AD) scores and labels using k-NN distances.

    Steps:
    - Fit a NearestNeighbors model on X_train.
    - For each sample, compute the mean distance to its k nearest neighbours.
    - Normalise distances relative to the distribution of training distances.
    - Classify samples with quantile thresholds.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ad_scores : float in [0, 1]
        ad_labels : str in {"in_domain", "borderline", "out_of_domain"}
    """
    if X_train is None or X_new is None:
        raise ValueError("X_train and X_new must be provided.")

    X_train_arr = np.asarray(X_train, dtype=float)
    X_new_arr = np.asarray(X_new, dtype=float)

    if X_train_arr.ndim != 2 or X_new_arr.ndim != 2:
        raise ValueError("X_train and X_new must be 2D arrays.")
    if X_train_arr.shape[1] != X_new_arr.shape[1]:
        raise ValueError("X_train and X_new must have the same number of features.")
    if X_train_arr.shape[0] == 0 or X_new_arr.shape[0] == 0:
        raise ValueError("X_train and X_new must be non-empty.")

    k = max(1, min(int(n_neighbors), X_train_arr.shape[0]))
    k_train = min(X_train_arr.shape[0], k + 1)  # +1 to drop self distances
    knn = NearestNeighbors(n_neighbors=k_train)
    knn.fit(X_train_arr)

    # Distances inside training set (exclude self if possible)
    train_distances, _ = knn.kneighbors(X_train_arr, n_neighbors=k_train, return_distance=True)
    if train_distances.shape[1] > 1:
        train_distances = train_distances[:, 1:]
    train_mean = train_distances.mean(axis=1)

    # Distances for new samples vs training set
    k_new = min(k, X_train_arr.shape[0])
    new_distances, _ = knn.kneighbors(X_new_arr, n_neighbors=k_new, return_distance=True)
    new_mean = new_distances.mean(axis=1)

    q_low = np.quantile(train_mean, quantiles[0])
    q_high = np.quantile(train_mean, quantiles[1])

    d_min = float(train_mean.min())
    d_max = float(train_mean.max())
    ad_scores = (new_mean - d_min) / (d_max - d_min + 1e-8)
    ad_scores = np.clip(ad_scores, 0.0, 1.0)

    ad_labels: np.ndarray = np.empty(len(new_mean), dtype=object)
    ad_labels[new_mean <= q_low] = "in_domain"
    mask_borderline = (new_mean > q_low) & (new_mean <= q_high)
    ad_labels[mask_borderline] = "borderline"
    ad_labels[new_mean > q_high] = "out_of_domain"

    return ad_scores, ad_labels


def compute_ad_for_dataframe(
    df_train: pd.DataFrame,
    df_new: pd.DataFrame,
    feature_cols: List[str],
    n_neighbors: int = 5,
    quantiles: Tuple[float, float] = (0.80, 0.95),
) -> pd.DataFrame:
    """
    Compute AD scores and labels for df_new given df_train and the feature columns.

    Returns a copy of df_new with added columns:
    - 'ad_score' (float in [0, 1])
    - 'ad_label' (str: "in_domain", "borderline", "out_of_domain")
    """
    if not feature_cols:
        raise ValueError("feature_cols must not be empty.")
    X_train = df_train[feature_cols].to_numpy(dtype=float)
    X_new = df_new[feature_cols].to_numpy(dtype=float)
    ad_scores, ad_labels = compute_knn_applicability_domain(
        X_train, X_new, n_neighbors=n_neighbors, quantiles=quantiles
    )
    out = df_new.copy()
    out["ad_score"] = ad_scores
    out["ad_label"] = ad_labels
    return out
