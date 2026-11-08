"""Streamlit dashboard for evaluating the five pre-trained classifiers."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation import METRIC_COLUMNS, evaluate_model
from src.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, validate_columns


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(
    page_title="ML Classification Studio | BITS Pilani M.Tech AIML",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);}
    [data-testid="stSidebar"] {background: #0f172a;}
    [data-testid="stSidebar"] * {color: #f8fafc;}
    .hero {padding: 1.5rem 1.8rem; border-radius: 18px; color: white;
           background: linear-gradient(120deg, #172554, #3730a3 55%, #0e7490);
           box-shadow: 0 12px 30px rgba(30, 41, 59, .16); margin-bottom: 1rem;}
    .hero h1 {margin: 0; font-size: 2.1rem;}
    .hero p {margin: .55rem 0 0; color: #dbeafe;}
    .hint {background: white; padding: .85rem 1rem; border-left: 4px solid #4f46e5;
           border-radius: 8px; color: #334155;}
    [data-testid="stMetric"] {background: white; border: 1px solid #e2e8f0;
           padding: .8rem; border-radius: 12px; box-shadow: 0 4px 12px rgba(15,23,42,.05);}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_models() -> dict[str, object]:
    models = {}
    for display_name, filename in MODEL_FILES.items():
        path = PROJECT_ROOT / "models" / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing model file: {path.name}. Run train_models.py first.")
        models[display_name] = joblib.load(path)
    return models


@st.cache_data(show_spinner=False)
def load_default_test_data() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "test_data.csv")


def confusion_figure(matrix: np.ndarray, labels: list[str], title: str):
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="Actual class",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#172554",
                fontsize=8,
            )
    figure.tight_layout()
    return figure


def prepare_uploaded_data(uploaded_file) -> pd.DataFrame:
    try:
        frame = pd.read_csv(uploaded_file)
    except Exception as exc:
        raise ValueError(f"The uploaded file could not be read as CSV: {exc}") from exc
    validate_columns(frame, require_target=True)
    prepared = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    if prepared[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")
    return prepared


st.markdown(
    """
    <div class="hero">
      <h1>ML Classification Studio</h1>
      <p>BITS Pilani M.Tech AIML | Dry Bean multiclass classification dashboard</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Evaluation controls")
    st.caption("Models are loaded from saved joblib pipelines. No training occurs in this app.")
    uploaded = st.file_uploader("Upload labelled test CSV", type=["csv"])
    selected_model = st.selectbox("Select a model", list(MODEL_FILES))
    compare_all = st.toggle("Compare all five models", value=False)
    st.divider()
    st.markdown("**Required schema**")
    st.caption(f"{len(FEATURE_COLUMNS)} input features plus `{TARGET_COLUMN}`.")
    st.download_button(
        "Download sample test CSV",
        data=(PROJECT_ROOT / "test_data.csv").read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

try:
    models = load_models()
except Exception as exc:
    st.error(str(exc))
    st.stop()

try:
    evaluation_data = prepare_uploaded_data(uploaded) if uploaded else load_default_test_data()
except ValueError as exc:
    st.error(str(exc))
    st.info("Use the sample CSV from the sidebar as a schema template.")
    st.stop()

source_label = "Uploaded CSV" if uploaded else "Bundled held-out test set"
st.markdown(
    f'<div class="hint"><b>Active data:</b> {source_label} - '
    f'{len(evaluation_data):,} rows, {len(FEATURE_COLUMNS)} features, target <b>{TARGET_COLUMN}</b>.</div>',
    unsafe_allow_html=True,
)

preview_tab, evaluation_tab, comparison_tab = st.tabs(
    ["Dataset preview", "Selected model", "All-model comparison"]
)

with preview_tab:
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Data preview")
        st.dataframe(evaluation_data.head(25), use_container_width=True, hide_index=True)
    with right:
        st.subheader("Class distribution")
        distribution = evaluation_data[TARGET_COLUMN].value_counts().sort_index()
        st.bar_chart(distribution, color="#4f46e5")
        st.caption(f"Missing feature values: {int(evaluation_data[FEATURE_COLUMNS].isna().sum().sum()):,}")

x_test = evaluation_data[FEATURE_COLUMNS]
y_test = evaluation_data[TARGET_COLUMN].astype(str)

with evaluation_tab:
    st.subheader(selected_model)
    result = evaluate_model(models[selected_model], x_test, y_test)
    metric_columns = st.columns(6)
    for column, metric in zip(metric_columns, METRIC_COLUMNS):
        value = result["metrics"][metric]
        column.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

    chart_column, report_column = st.columns([1.05, 1])
    with chart_column:
        st.pyplot(
            confusion_figure(
                result["confusion_matrix"],
                result["labels"],
                f"{selected_model} - Confusion Matrix",
            ),
            use_container_width=True,
        )
    with report_column:
        st.markdown("#### Classification report")
        report_frame = pd.DataFrame(result["classification_report"]).T
        st.dataframe(report_frame.style.format("{:.4f}"), use_container_width=True)

    st.markdown("#### Actual versus predicted class summary")
    summary = pd.crosstab(
        pd.Series(y_test.to_numpy(), name="Actual"),
        pd.Series(result["predictions"], name="Predicted"),
        margins=True,
    )
    st.dataframe(summary, use_container_width=True)

    predictions = evaluation_data.copy()
    predictions["Predicted_Class"] = result["predictions"]
    predictions["Correct"] = predictions[TARGET_COLUMN].astype(str) == predictions["Predicted_Class"].astype(str)
    st.download_button(
        "Download predictions",
        data=predictions.to_csv(index=False).encode("utf-8"),
        file_name=f"{MODEL_FILES[selected_model].replace('.joblib', '')}_predictions.csv",
        mime="text/csv",
    )

with comparison_tab:
    if compare_all:
        with st.spinner("Evaluating all saved models on the active test data..."):
            comparison_rows = []
            for name, model in models.items():
                evaluated = evaluate_model(model, x_test, y_test)
                comparison_rows.append({"Model": name, **evaluated["metrics"]})
            comparison = pd.DataFrame(comparison_rows).set_index("Model")
        st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0), use_container_width=True)
        st.bar_chart(comparison[METRIC_COLUMNS], height=430)
        st.caption("Higher is better for every displayed metric.")
    else:
        st.info("Enable **Compare all five models** in the sidebar to run this view.")

st.caption("Academic project - predictions are for demonstration and evaluation only.")
