"""Consistent multiclass evaluation helpers shared by training and Streamlit."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]


def _multiclass_auc(model: Any, features: pd.DataFrame, target: pd.Series) -> float:
    if not hasattr(model, "predict_proba"):
        return float("nan")
    probabilities = model.predict_proba(features)
    classes = np.asarray(model.classes_)
    binary_target = label_binarize(target, classes=classes)
    if binary_target.shape[1] != probabilities.shape[1]:
        raise ValueError("Probability output does not match the model class labels.")
    return float(roc_auc_score(binary_target, probabilities, average="macro", multi_class="ovr"))


def evaluate_model(model: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, Any]:
    predictions = model.predict(features)
    try:
        auc_value = _multiclass_auc(model, features, target)
    except (ValueError, AttributeError):
        auc_value = float("nan")

    labels = [str(label) for label in model.classes_]
    metrics = {
        "Accuracy": float(accuracy_score(target, predictions)),
        "AUC": auc_value,
        "Precision": float(precision_score(target, predictions, average="weighted", zero_division=0)),
        "Recall": float(recall_score(target, predictions, average="weighted", zero_division=0)),
        "F1 Score": float(f1_score(target, predictions, average="weighted", zero_division=0)),
        "MCC": float(matthews_corrcoef(target, predictions)),
    }
    report = classification_report(
        target,
        predictions,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(target, predictions, labels=labels)
    return {
        "metrics": metrics,
        "predictions": predictions,
        "labels": labels,
        "classification_report": report,
        "confusion_matrix": matrix,
    }


def rank_models(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank models by their mean rank across all six metrics; lower is better."""
    ranked = comparison.copy()
    rank_columns = []
    for metric in METRIC_COLUMNS:
        column = f"{metric} Rank"
        ranked[column] = ranked[metric].rank(method="average", ascending=False)
        rank_columns.append(column)
    ranked["Mean Rank"] = ranked[rank_columns].mean(axis=1)
    return ranked.sort_values(["Mean Rank", "F1 Score", "MCC"], ascending=[True, False, False])
