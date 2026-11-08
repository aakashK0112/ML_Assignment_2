"""Dataset validation, inspection, and leakage-safe splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


TARGET_COLUMN = "Class"
FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]


@dataclass(frozen=True)
class DatasetAudit:
    rows: int
    columns: int
    missing_values: int
    duplicate_rows: int
    class_distribution: dict[str, int]
    feature_dtypes: dict[str, str]


def audit_dataset(data: pd.DataFrame) -> DatasetAudit:
    """Return the checks required by the assignment without mutating data."""
    validate_columns(data, require_target=True)
    return DatasetAudit(
        rows=int(data.shape[0]),
        columns=int(data.shape[1]),
        missing_values=int(data.isna().sum().sum()),
        duplicate_rows=int(data.duplicated().sum()),
        class_distribution={
            str(label): int(count)
            for label, count in data[TARGET_COLUMN].value_counts().sort_index().items()
        },
        feature_dtypes={column: str(data[column].dtype) for column in FEATURE_COLUMNS},
    )


def validate_columns(data: pd.DataFrame, require_target: bool = True) -> None:
    required = set(FEATURE_COLUMNS)
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, coerce features to numeric, and remove exact duplicates."""
    validate_columns(data, require_target=True)
    cleaned = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype("string").str.strip()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    if cleaned[TARGET_COLUMN].isna().any() or (cleaned[TARGET_COLUMN] == "").any():
        raise ValueError("Target column contains missing or blank labels.")
    return cleaned


def stratified_split(data: pd.DataFrame):
    """Create the fixed 80:20 split used by every model."""
    features = data[FEATURE_COLUMNS]
    target = data[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
