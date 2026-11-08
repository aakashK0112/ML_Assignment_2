"""Train, evaluate, and persist all assignment models.

Run from any directory with: python path/to/ML_Assignment_2/train_models.py
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation import METRIC_COLUMNS, evaluate_model, rank_models
from src.model_training import build_model_pipelines
from src.preprocessing import TARGET_COLUMN, audit_dataset, clean_dataset, stratified_split


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFUSION_DIR = RESULTS_DIR / "confusion_matrices"
ASSETS_DIR = PROJECT_ROOT / "assets"


def ensure_directories() -> None:
    for directory in (MODELS_DIR, RESULTS_DIR, CONFUSION_DIR, ASSETS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], title: str, stem: str) -> None:
    table = pd.DataFrame(matrix, index=labels, columns=labels)
    table.to_csv(CONFUSION_DIR / f"{stem}.csv", index_label="Actual / Predicted")

    figure, axis = plt.subplots(figsize=(8.3, 7.0))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="Actual class",
        title=f"{title} - Confusion Matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:d}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#172554",
                fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(CONFUSION_DIR / f"{stem}.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_comparison_chart(comparison: pd.DataFrame) -> None:
    plot_data = comparison.set_index("Model")[METRIC_COLUMNS]
    axis = plot_data.plot(kind="bar", figsize=(13, 6.8), width=0.82, colormap="viridis")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Held-out Test Performance Across Six Evaluation Metrics", pad=14, weight="bold")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.legend(loc="lower center", bbox_to_anchor=(0.5, -0.37), ncol=3, frameon=False)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "performance_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    ensure_directories()
    raw_data = pd.read_csv(DATASET_PATH)
    audit = audit_dataset(raw_data)
    clean_data = clean_dataset(raw_data)
    x_train, x_test, y_train, y_test = stratified_split(clean_data)

    test_data = x_test.copy()
    test_data[TARGET_COLUMN] = y_test
    test_data.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

    model_rows: list[dict[str, float | str]] = []
    reports: dict[str, dict] = {}

    for display_name, (stem, pipeline) in build_model_pipelines().items():
        print(f"Training {display_name}...")
        pipeline.fit(x_train, y_train)
        joblib.dump(pipeline, MODELS_DIR / f"{stem}.joblib", compress=3)

        evaluation = evaluate_model(pipeline, x_test, y_test)
        model_rows.append({"Model": display_name, **evaluation["metrics"]})
        reports[display_name] = evaluation["classification_report"]
        save_confusion_matrix(
            evaluation["confusion_matrix"],
            evaluation["labels"],
            display_name,
            stem,
        )

    comparison = pd.DataFrame(model_rows)
    ranked = rank_models(comparison)
    comparison = comparison.merge(ranked[["Model", "Mean Rank"]], on="Model", how="left")
    comparison = comparison.sort_values("Mean Rank").reset_index(drop=True)
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False, float_format="%.6f")

    with (RESULTS_DIR / "classification_reports.json").open("w", encoding="utf-8") as handle:
        json.dump(reports, handle, indent=2)

    metadata = {
        "source": "UCI Machine Learning Repository - Dry Bean dataset",
        "source_url": "https://archive.ics.uci.edu/dataset/602/dry+bean",
        "doi": "10.24432/C50S4B",
        "raw_audit": asdict(audit),
        "clean_rows": int(clean_data.shape[0]),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "random_state": 42,
        "winner": str(comparison.iloc[0]["Model"]),
        "winner_mean_rank": float(comparison.iloc[0]["Mean Rank"]),
    }
    with (RESULTS_DIR / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    save_comparison_chart(comparison)
    print("\nModel comparison (best balanced rank first):")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nWinning model: {metadata['winner']}")
    print(f"Saved held-out test set: {PROJECT_ROOT / 'test_data.csv'}")


if __name__ == "__main__":
    main()
