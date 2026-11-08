"""Model definitions with preprocessing embedded inside every saved pipeline."""

from __future__ import annotations

from collections import OrderedDict

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import FEATURE_COLUMNS


def _numeric_preprocessor(scale: bool) -> ColumnTransformer:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps)
    return ColumnTransformer(
        [("numeric", numeric_pipeline, FEATURE_COLUMNS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipelines() -> OrderedDict[str, tuple[str, Pipeline]]:
    """Return display name -> (filename stem, fitted-ready pipeline)."""
    specifications = OrderedDict(
        [
            (
                "Logistic Regression",
                (
                    "logistic_regression",
                    LogisticRegression(max_iter=2500, C=2.0, solver="lbfgs", random_state=42),
                    True,
                ),
            ),
            (
                "Decision Tree",
                (
                    "decision_tree",
                    DecisionTreeClassifier(
                        max_depth=14,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=42,
                    ),
                    False,
                ),
            ),
            ("KNN", ("knn", KNeighborsClassifier(n_neighbors=9, weights="distance", p=2), True)),
            ("Naive Bayes", ("naive_bayes", GaussianNB(var_smoothing=1e-9), True)),
            (
                "Support Vector Machine",
                (
                    "svm",
                    SVC(C=4.0, kernel="rbf", gamma="scale", probability=True, random_state=42),
                    True,
                ),
            ),
        ]
    )

    pipelines: OrderedDict[str, tuple[str, Pipeline]] = OrderedDict()
    for display_name, (stem, estimator, scale) in specifications.items():
        pipelines[display_name] = (
            stem,
            Pipeline(
                [
                    ("preprocessor", _numeric_preprocessor(scale=scale)),
                    ("classifier", estimator),
                ]
            ),
        )
    return pipelines
