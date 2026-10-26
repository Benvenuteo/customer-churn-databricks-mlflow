"""Metrics and uncertainty calculated from bounded held-out prediction tables."""

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(
    y: np.ndarray, probability: np.ndarray, threshold: float = 0.5, seed: int = 42
) -> dict[str, Any]:
    """Use AP for the precision-recall summary and seeded tie-breaking for top-decile ranking."""
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if not len(y) or len(y) != len(probability):
        raise ValueError("Metrics require nonempty, equally sized labels and predictions")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("Classification labels must be binary")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    predicted = probability >= threshold
    permutation = np.random.default_rng(seed).permutation(len(y))
    order = permutation[np.argsort(-probability[permutation], kind="stable")]
    top_count = max(1, math.ceil(0.1 * len(y)))
    selected = y[order[:top_count]]
    prevalence = float(y.mean())
    top_precision = float(selected.mean())
    return {
        "rows": len(y),
        "prevalence": prevalence,
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, probability)) if y.sum() else None,
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.clip(probability, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, predicted, labels=[0, 1])
        .ravel()
        .tolist(),
        "precision_at_top_10pct": top_precision,
        "recall_at_top_10pct": float(selected.sum() / y.sum()) if y.sum() else None,
        "lift_at_top_10pct": top_precision / prevalence if prevalence else None,
    }


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Report errors in the dataset's spending units; no currency is assumed."""
    if not len(y) or len(y) != len(prediction):
        raise ValueError("Metrics require nonempty, equally sized labels and predictions")
    if not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Regression labels and predictions must be finite")
    return {
        "rows": len(y),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)) if len(y) > 1 and np.var(y) > 0 else None,
    }


def choose_threshold(y: np.ndarray, probability: np.ndarray) -> float:
    """Choose F1 on validation only; break ties toward the conventional 0.5 threshold."""
    candidates = np.linspace(0.05, 0.95, 91)
    scored = [
        (
            f1_score(y, probability >= threshold, zero_division=0),
            -abs(threshold - 0.5),
            float(threshold),
        )
        for threshold in candidates
    ]
    return max(scored)[2]


def bootstrap_intervals(
    y: np.ndarray, prediction: np.ndarray, task: str, repeats: int, seed: int
) -> dict[str, Any]:
    """Percentile CIs condition on the fitted model; classification resampling preserves class counts."""
    if repeats < 1:
        return {"repeats": 0, "intervals": {}}
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {}
    for _ in range(repeats):
        if task == "classification":
            groups = [np.flatnonzero(y == value) for value in (0, 1)]
            if any(len(group) == 0 for group in groups):
                return {"repeats": 0, "intervals": {}, "reason": "Both classes are required"}
            selected = np.concatenate(
                [rng.choice(group, len(group), replace=True) for group in groups]
            )
            scores = {
                "roc_auc": roc_auc_score(y[selected], prediction[selected]),
                "average_precision": average_precision_score(y[selected], prediction[selected]),
            }
        else:
            selected = rng.integers(0, len(y), len(y))
            scores = {
                "mae": mean_absolute_error(y[selected], prediction[selected]),
                "rmse": mean_squared_error(y[selected], prediction[selected]) ** 0.5,
            }
        for key, score in scores.items():
            values.setdefault(key, []).append(float(score))
    return {
        "repeats": repeats,
        "confidence": 0.95,
        "method": "Percentile bootstrap of fixed held-out predictions; no training uncertainty",
        "intervals": {
            key: {"low": float(np.quantile(value, 0.025)), "high": float(np.quantile(value, 0.975))}
            for key, value in values.items()
        },
    }


def segment_errors(predictions: pd.DataFrame, task: str, threshold: float) -> list[dict[str, Any]]:
    """Report sufficiently populated predefined customer segments, without model reselection."""
    results = []
    for column in ("gender", "contract_length"):
        for name, group in predictions.groupby(column, dropna=False):
            if len(group) < 20:
                continue
            y = group["label"].to_numpy()
            values = group["score"].to_numpy()
            metrics = (
                classification_metrics(y, values, threshold)
                if task == "classification"
                else regression_metrics(y, values)
            )
            results.append({"feature": column, "value": str(name), **metrics})
    return results
