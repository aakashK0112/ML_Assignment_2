# ML Classification Studio - BITS Pilani M.Tech AIML

## 1. Problem Statement

This project builds an end-to-end multiclass classification system for identifying seven dry bean varieties from 16 geometric and shape features. Five classifiers are trained on the same leakage-safe 80:20 stratified split and evaluated on six complementary metrics.

## 2. Dataset Description

- **Dataset:** UCI Dry Bean
- **Raw shape:** 13,611 rows x 17 columns
- **Features:** 16 numeric predictors
- **Classes:** 7 (BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA)
- **Missing values:** 0
- **Exact duplicates detected and removed before splitting:** 68
- **Clean rows:** 13,543
- **Split:** 10,834 training rows and 2,709 held-out test rows

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

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC | Mean Rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| Support Vector Machine | 0.9236 | 0.9946 | 0.9237 | 0.9236 | 0.9236 | 0.9075 | 1.1667 |
| Logistic Regression | 0.9210 | 0.9948 | 0.9216 | 0.9210 | 0.9211 | 0.9045 | 1.8333 |
| KNN | 0.9133 | 0.9872 | 0.9142 | 0.9133 | 0.9134 | 0.8951 | 3.1667 |
| Naive Bayes | 0.8970 | 0.9911 | 0.8997 | 0.8970 | 0.8972 | 0.8762 | 3.8333 |
| Decision Tree | 0.8907 | 0.9589 | 0.8915 | 0.8907 | 0.8910 | 0.8680 | 5.0000 |

## 9. Observations on Each Model

- **Logistic Regression:** Strong linear baseline with 0.9210 accuracy and 0.9948 macro AUC. Its high AUC shows excellent class separation, while its F1 and MCC remain close to the best model.
- **Decision Tree:** The most interpretable nonlinear model, but the lowest balanced result: 0.8910 weighted F1 and 0.8680 MCC. The gap suggests higher variance than the scaled models.
- **KNN:** Distance-weighted KNN reached 0.9133 accuracy and 0.8951 MCC after standardization, giving a solid nonlinear result without an explicit parametric boundary.
- **Naive Bayes:** Gaussian Naive Bayes achieved 0.9911 macro AUC, but its 0.8972 F1 score trails the top models. Correlated geometric features make its conditional-independence assumption restrictive.
- **Support Vector Machine:** Best balanced performer with 0.9236 accuracy, 0.9236 F1, and 0.9075 MCC. The RBF kernel captures nonlinear class boundaries while retaining excellent probability-based AUC.

## 10. Overall Winning Model

**Support Vector Machine** is the overall winner. Models are ranked independently on all six metrics, then ordered by mean rank with weighted F1 and MCC as tie-breakers. This avoids choosing by accuracy alone. The winning mean rank is **1.1667**.

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
.venv\Scripts\activate
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

Streamlit URL: [https://mlassignment2-q636a5dubby6qwtjazq9oa.streamlit.app](https://mlassignment2-q636a5dubby6qwtjazq9oa.streamlit.app)

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

- Replace the student-details and BITS screenshot placeholders before submission.
- Do not upload private credentials or the `.venv` folder to GitHub.
- Results are reproducible from the bundled dataset with `random_state=42`.
