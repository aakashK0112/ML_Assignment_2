"""Generate the metrics-backed README and final PDF report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PDF_PATH = ROOT / "BITS_MTech_AIML_ML_Assignment_2.pdf"
README_PATH = ROOT / "README.md"


def format_table_markdown(comparison: pd.DataFrame) -> str:
    headers = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC", "Mean Rank"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * 7) + "|"]
    for _, row in comparison.iterrows():
        lines.append(
            "| " + " | ".join(
                [str(row["Model"])] + [f"{float(row[column]):.4f}" for column in headers[1:]]
            ) + " |"
        )
    return "\n".join(lines)


def observations(comparison: pd.DataFrame) -> dict[str, str]:
    by_name = comparison.set_index("Model")
    return {
        "Logistic Regression": (
            f"Strong linear baseline with {by_name.loc['Logistic Regression', 'Accuracy']:.4f} accuracy and "
            f"{by_name.loc['Logistic Regression', 'AUC']:.4f} macro AUC. Its high AUC shows excellent class "
            "separation, while its F1 and MCC remain close to the best model."
        ),
        "Decision Tree": (
            f"The most interpretable nonlinear model, but the lowest balanced result: "
            f"{by_name.loc['Decision Tree', 'F1 Score']:.4f} weighted F1 and "
            f"{by_name.loc['Decision Tree', 'MCC']:.4f} MCC. The gap suggests higher variance than the scaled models."
        ),
        "KNN": (
            f"Distance-weighted KNN reached {by_name.loc['KNN', 'Accuracy']:.4f} accuracy and "
            f"{by_name.loc['KNN', 'MCC']:.4f} MCC after standardization, giving a solid nonlinear result without "
            "an explicit parametric boundary."
        ),
        "Naive Bayes": (
            f"Gaussian Naive Bayes achieved {by_name.loc['Naive Bayes', 'AUC']:.4f} macro AUC, but its "
            f"{by_name.loc['Naive Bayes', 'F1 Score']:.4f} F1 score trails the top models. Correlated geometric "
            "features make its conditional-independence assumption restrictive."
        ),
        "Support Vector Machine": (
            f"Best balanced performer with {by_name.loc['Support Vector Machine', 'Accuracy']:.4f} accuracy, "
            f"{by_name.loc['Support Vector Machine', 'F1 Score']:.4f} F1, and "
            f"{by_name.loc['Support Vector Machine', 'MCC']:.4f} MCC. The RBF kernel captures nonlinear class "
            "boundaries while retaining excellent probability-based AUC."
        ),
    }


def build_readme(comparison: pd.DataFrame, metadata: dict) -> None:
    notes = observations(comparison)
    winner = metadata["winner"]
    content = f"""# ML Classification Studio - BITS Pilani M.Tech AIML

## 1. Problem Statement

This project builds an end-to-end multiclass classification system for identifying seven dry bean varieties from 16 geometric and shape features. Five classifiers are trained on the same leakage-safe 80:20 stratified split and evaluated on six complementary metrics.

## 2. Dataset Description

- **Dataset:** UCI Dry Bean
- **Raw shape:** {metadata['raw_audit']['rows']:,} rows x {metadata['raw_audit']['columns']} columns
- **Features:** {metadata['feature_count']} numeric predictors
- **Classes:** 7 (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA)
- **Missing values:** {metadata['raw_audit']['missing_values']}
- **Exact duplicates detected and removed before splitting:** {metadata['raw_audit']['duplicate_rows']}
- **Clean rows:** {metadata['clean_rows']:,}
- **Split:** {metadata['train_rows']:,} training rows and {metadata['test_rows']:,} held-out test rows

## 3. Dataset Source

UCI Machine Learning Repository: [Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean), DOI: [10.24432/C50S4B](https://doi.org/10.24432/C50S4B). The dataset is licensed CC BY 4.0.

## 4. GitHub Repository Link

[https://github.com/aakashK0112/ML_Assignment_2](https://github.com/aakashK0112/ML_Assignment_2)

## 5. Data Preprocessing

The schema is validated, feature columns are coerced to numeric, and exact duplicate records are removed before splitting. An 80:20 stratified split with `random_state=42` is used. Median imputation is fitted only on the training fold. Logistic Regression, KNN, Gaussian Naive Bayes, and SVM also use training-fitted `StandardScaler`; Decision Tree uses median imputation without scaling. Each preprocessing chain is embedded in its saved scikit-learn `Pipeline`, preventing leakage when uploaded data is evaluated.

## 6. Machine Learning Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. Distance-weighted K-Nearest Neighbors
4. Gaussian Naive Bayes
5. RBF Support Vector Machine with probability estimates

## 7. Model Evaluation Metrics

Accuracy, multiclass one-vs-rest macro AUC, weighted Precision, weighted Recall, weighted F1 Score, and Matthews Correlation Coefficient (MCC) are calculated on the same held-out test set.

## 8. Model Comparison Table

{format_table_markdown(comparison)}

## 9. Observations on Each Model

""" + "\n".join(f"- **{model}:** {text}" for model, text in notes.items()) + f"""

## 10. Overall Winning Model

**{winner}** is the overall winner. Models are ranked independently on all six metrics, then ordered by mean rank with weighted F1 and MCC as tie-breakers. This avoids choosing by accuracy alone. The winning mean rank is **{metadata['winner_mean_rank']:.4f}**.

## 11. Streamlit Application Features

- Labelled CSV upload with required-column validation and clear errors
- Model selection across all five persisted pipelines
- Six evaluation metrics, confusion matrix, classification report, and actual-vs-predicted table
- Optional all-model comparison on the uploaded test data
- Dataset preview, class distribution, sample CSV, and prediction download
- No model fitting or preprocessing fitting inside the app

## 12. Installation Instructions

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 13. How to Run Locally

The folder already contains trained models and a held-out test file, so the shortest command is:

```bash
streamlit run app.py
```

To reproduce every model and regenerate all evidence:

```bash
python train_models.py
python generate_report.py
streamlit run app.py
```

On a Windows BITS Virtual Lab machine, double-click `RUN_ON_BITS_MACHINE.bat` for setup and launch.

## 14. Deployment on Streamlit Community Cloud

1. Create a new GitHub repository and upload the complete contents of this folder.
2. Confirm that the five `.joblib` files, `test_data.csv`, and `requirements.txt` are committed.
3. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, and select the repository.
4. Set the main file path to `app.py` and deploy.
5. Replace the placeholders below only after URLs exist.

Streamlit URL: `<ADD_YOUR_STREAMLIT_DEPLOYMENT_URL_HERE>`

## 15. Project Folder Structure

```text
ML_Assignment_2/
|-- app.py
|-- train_models.py
|-- generate_report.py
|-- dataset.csv
|-- test_data.csv
|-- requirements.txt
|-- README.md
|-- BITS_MTech_AIML_ML_Assignment_2.pdf
|-- RUN_ON_BITS_MACHINE.bat
|-- models/                  # five trained pipelines
|-- results/                 # metrics, reports, audit, confusion matrices
|-- src/                     # preprocessing, training, evaluation modules
`-- assets/                  # performance chart
```

## 16. Conclusion

All five models were trained and compared under identical split conditions. The RBF SVM provides the strongest overall balance, while Logistic Regression is a very competitive and simpler baseline. The application evaluates only persisted pipelines and never learns from uploaded test data.

## Academic and Reproducibility Notes

- Replace student, GitHub, Streamlit, and BITS screenshot placeholders before submission.
- Do not upload private credentials or the `.venv` folder to GitHub.
- Results are reproducible from the bundled dataset with `random_state=42`.
"""
    README_PATH.write_text(content, encoding="utf-8")


def _draw_running_furniture(pdf_canvas, page_number: int) -> None:
    pdf_canvas.saveState()
    width, height = pdf_canvas._pagesize
    pdf_canvas.setFillColor(colors.HexColor("#172554"))
    pdf_canvas.rect(0, height - 13 * mm, width, 13 * mm, fill=1, stroke=0)
    pdf_canvas.setFillColor(colors.white)
    pdf_canvas.setFont("Helvetica-Bold", 8.5)
    pdf_canvas.drawString(15 * mm, height - 8.5 * mm, "BITS Pilani M.Tech AIML - Machine Learning Assignment 2")
    pdf_canvas.setFillColor(colors.HexColor("#475569"))
    pdf_canvas.setFont("Helvetica", 8)
    pdf_canvas.drawRightString(width - 15 * mm, 9 * mm, f"Page {page_number}")
    pdf_canvas.restoreState()


class ReportCanvas(canvas.Canvas):
    """Draw running furniture after flowables for reliable PDF viewers."""

    def showPage(self) -> None:  # noqa: N802 - ReportLab API name
        _draw_running_furniture(self, self._pageNumber)
        super().showPage()


def _header_footer_on_page(pdf_canvas, document) -> None:
    # Some ReportLab page-break paths paint after showPage furniture. Drawing in
    # both lifecycle hooks keeps headers reliable across PDF renderers.
    _draw_running_furniture(pdf_canvas, document.page)


def build_pdf(comparison: pd.DataFrame, metadata: dict) -> None:
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=colors.HexColor("#172554"), alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="SubTitle", parent=styles["Normal"], fontSize=12, leading=17, textColor=colors.HexColor("#475569"), alignment=TA_CENTER, spaceAfter=15))
    styles.add(ParagraphStyle(name="H1Custom", parent=styles["Heading1"], fontSize=16, leading=20, textColor=colors.HexColor("#1e3a8a"), spaceBefore=8, spaceAfter=7))
    styles.add(ParagraphStyle(name="H2Custom", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#3730a3"), spaceBefore=6, spaceAfter=4))
    styles.add(ParagraphStyle(name="BodyCustom", parent=styles["BodyText"], fontSize=9.2, leading=13.2, textColor=colors.HexColor("#1e293b"), spaceAfter=6))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#334155")))

    document = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=landscape(A4),
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=20 * mm,
        bottomMargin=15 * mm,
        title="BITS M.Tech AIML Machine Learning Assignment 2",
        author="Student Name Placeholder",
    )
    story = []
    body = styles["BodyCustom"]
    h1 = styles["H1Custom"]
    h2 = styles["H2Custom"]

    story.extend([
        Spacer(1, 17 * mm),
        Paragraph("Machine Learning Assignment 2", styles["TitleCustom"]),
        Paragraph("End-to-End Multiclass Classification of Dry Bean Varieties", styles["SubTitle"]),
        Spacer(1, 7 * mm),
        Table(
            [
                ["Student Name", "<ADD STUDENT NAME>"],
                ["Student ID", "<ADD BITS ID>"],
                ["Programme", "M.Tech Artificial Intelligence and Machine Learning"],
                ["Course", "Machine Learning"],
                ["GitHub Repository", "https://github.com/aakashK0112/ML_Assignment_2"],
                ["Streamlit Application", "<ADD STREAMLIT DEPLOYMENT URL>"],
            ],
            colWidths=[48 * mm, 150 * mm],
            style=TableStyle([
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0e7ff")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1e3a8a")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("GRID", (0, 0), (-1, -1), .5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]),
        ),
        Spacer(1, 10 * mm),
        Paragraph("Prepared as an original, reproducible academic submission. Experimental values in this report were calculated from the bundled data and saved models.", styles["SubTitle"]),
        PageBreak(),
        Paragraph("1. Problem Statement", h1),
        Paragraph("The objective is to classify dry bean grains into seven registered varieties using 16 geometric and shape measurements. Five standard classifiers are trained on the same training split, compared with six metrics, persisted with their preprocessing, and exposed through a Streamlit evaluation dashboard.", body),
        Paragraph("2. Dataset Description and Source", h1),
        Paragraph(f"The UCI Dry Bean dataset contains {metadata['raw_audit']['rows']:,} observations, 16 numeric input features, and the target Class. It represents BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, and SIRA. The raw file contains {metadata['raw_audit']['missing_values']} missing values and {metadata['raw_audit']['duplicate_rows']} exact duplicates. After duplicate removal, {metadata['clean_rows']:,} rows remain.", body),
        Paragraph("Source: UCI Machine Learning Repository, Dry Bean dataset. DOI: 10.24432/C50S4B. License: CC BY 4.0. URL: https://archive.ics.uci.edu/dataset/602/dry+bean", body),
        Paragraph("3. Preprocessing and Experimental Design", h1),
        Paragraph(f"Exact duplicates are removed before splitting. A stratified 80:20 split with random_state=42 produces {metadata['train_rows']:,} training records and {metadata['test_rows']:,} held-out test records. Median imputation is fit only on training data. StandardScaler is included for Logistic Regression, KNN, Gaussian Naive Bayes, and SVM; the Decision Tree is not scaled. Every transformer and estimator is stored in one scikit-learn Pipeline, preventing test-data leakage.", body),
        Paragraph("4. Model Implementation Summary", h1),
        Paragraph("Logistic Regression provides a regularized linear baseline. The Decision Tree captures nonlinear rules with depth and leaf constraints. KNN uses distance weighting after standardization. Gaussian Naive Bayes models per-class feature distributions. The RBF SVM uses nonlinear margins with probability=True for multiclass AUC.", body),
        PageBreak(),
        Paragraph("5. Model Comparison", h1),
        Paragraph("All scores below were calculated on the same held-out test set. AUC is macro one-vs-rest; Precision, Recall, and F1 are weighted averages.", body),
    ])

    headers = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC", "Mean Rank"]
    table_data = [headers]
    for _, row in comparison.iterrows():
        table_data.append([row["Model"]] + [f"{float(row[col]):.4f}" for col in ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC", "Mean Rank"]])
    metric_table = Table(table_data, colWidths=[48 * mm] + [24 * mm] * 7, repeatRows=1)
    metric_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dcfce7")),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.4),
        ("GRID", (0, 0), (-1, -1), .45, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 2), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([metric_table, PageBreak()])

    chart_path = ROOT / "assets" / "performance_comparison.png"
    if chart_path.exists():
        story.append(Image(str(chart_path), width=230 * mm, height=120 * mm))

    story.extend([PageBreak(), Paragraph("6. Model-wise Observations", h1)])
    for model, note in observations(comparison).items():
        story.append(KeepTogether([Paragraph(model, h2), Paragraph(note, body)]))
    winner = metadata["winner"]
    story.extend([
        Paragraph("7. Overall Winning Model", h1),
        Paragraph(f"{winner} is selected using a balanced rank across Accuracy, AUC, weighted Precision, weighted Recall, weighted F1, and MCC rather than accuracy alone. Its mean rank is {metadata['winner_mean_rank']:.4f}. The result reflects consistent discrimination, class-balanced predictive quality, and strong agreement beyond chance.", body),
        PageBreak(),
        Paragraph("8. Streamlit Application Features", h1),
        Paragraph("The application loads all five saved joblib pipelines and never retrains them. It validates an uploaded labelled CSV, previews the data, reports six metrics, renders a confusion matrix and classification report, summarizes actual versus predicted classes, compares all five models on the uploaded test set, and offers downloadable predictions. Invalid or incomplete files receive a clear error message.", body),
        Paragraph("9. Reproducibility and Quality Controls", h1),
        Paragraph("The included scripts recreate the fixed split, train every model, save all outputs, and regenerate this report. Model files contain preprocessing and estimator objects together. Source files use project-relative paths, making the folder portable across Windows and Streamlit Community Cloud. The held-out test CSV contains all 16 input features plus Class.", body),
        Paragraph("10. GitHub README Content Summary", h1),
        Paragraph("The repository README documents the problem, dataset and source, preprocessing, five models, metric definitions, actual comparison table, model observations, winning-model rationale, Streamlit features, installation, local execution, cloud deployment, folder structure, conclusion, and placeholders that must be replaced after deployment.", body),
        Paragraph("Repository: https://github.com/aakashK0112/ML_Assignment_2", body),
        Paragraph("Deployment placeholder: &lt;ADD STREAMLIT DEPLOYMENT URL&gt;", body),
        PageBreak(),
        Paragraph("11. BITS Virtual Lab Execution Evidence", h1),
        Paragraph("Run RUN_ON_BITS_MACHINE.bat or follow the README commands on the BITS machine. After Streamlit opens, capture the terminal showing the local URL and the dashboard showing the selected model metrics.", body),
        Spacer(1, 6 * mm),
        Table(
            [[Paragraph("INSERT BITS VIRTUAL LAB EXECUTION SCREENSHOT HERE<br/><br/>Suggested evidence: terminal command, successful Streamlit URL, dashboard metric cards, and model comparison view.", styles["SubTitle"])]],
            colWidths=[245 * mm],
            rowHeights=[95 * mm],
            style=TableStyle([
                ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#64748b")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ]),
        ),
        Spacer(1, 7 * mm),
        Paragraph("Before final submission, replace all student, GitHub, Streamlit, and screenshot placeholders. No URL or execution evidence has been fabricated.", body),
        PageBreak(),
        Paragraph("12. Conclusion", h1),
        Paragraph("The assignment delivers a reproducible multiclass classification workflow over a real UCI dataset. The RBF SVM is the strongest balanced model, with Logistic Regression a close and simpler alternative. The saved-pipeline architecture ensures that every local, uploaded, or cloud prediction receives exactly the transformations learned from training data.", body),
        Paragraph("13. Key Files", h1),
        Paragraph("app.py - Streamlit interface; train_models.py - reproducible training and evaluation; generate_report.py - README and PDF generation; dataset.csv - full UCI data; test_data.csv - held-out data; models/ - five saved pipelines; results/ - comparison, reports, audit, and confusion matrices; assets/ - performance chart.", body),
        Paragraph("14. Submission Checklist", h1),
        Paragraph("[ ] Add student name and BITS ID.  [ ] Create GitHub repository and replace its URL.  [ ] Deploy Streamlit and replace its URL.  [ ] Run on BITS Virtual Lab and insert an authentic screenshot.  [ ] Recheck the generated PDF and repository contents before upload.", body),
    ])

    document.build(
        story,
        onFirstPage=_header_footer_on_page,
        onLaterPages=_header_footer_on_page,
        canvasmaker=ReportCanvas,
    )


def main() -> None:
    comparison = pd.read_csv(RESULTS / "model_comparison.csv")
    metadata = json.loads((RESULTS / "dataset_audit.json").read_text(encoding="utf-8"))
    build_readme(comparison, metadata)
    build_pdf(comparison, metadata)
    print(f"Generated {README_PATH.name}")
    print(f"Generated {PDF_PATH.name}")


if __name__ == "__main__":
    main()
