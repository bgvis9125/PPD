# Postpartum Depression Risk Prediction using Machine Learning

## 1. Project Overview
Postpartum depression is a serious mental health condition that can affect mothers after childbirth. Early identification of high‑risk individuals can support timely intervention and better outcomes.

This project builds and compares multiple machine learning models to predict whether a patient is at **high risk** or **low risk** of postnatal depression based on clinical and demographic features. The best models and their behavior are further analyzed and visualized using **Tableau**.

Main goals:
- Build individual models (XGBoost, SVM, Neural Network).
- Build an **Ensemble Voting Classifier** combining these models.
- Evaluate models using multiple metrics (Accuracy, Precision, Recall, F1, ROC‑AUC).
- Analyze feature importance, model errors, and demographic patterns using Tableau dashboards.

## 2. Dataset

- File: `post-natal-data.csv`
- Each row represents a single patient record.
- Columns (examples – adjust to your actual features):
  - Demographic features: `Age`, `Marital_Status`, `Employment_Status`, etc.
  - Clinical / questionnaire features: scores or yes/no indicators related to mood, sleep, anxiety, etc.
  - Target label: e.g. `Risk` (values like `High_Risk` / `Low_Risk` or 1/0).

### 2.1 Data Preprocessing
Typical preprocessing steps used in the notebooks:
- Handling missing values (drop or impute).
- Encoding categorical variables (e.g. one‑hot encoding).
- Train–test split (e.g. 80% train / 20% test).
- Feature scaling where required (especially for SVM and Neural Network).


## 3. Models Implemented

All models are implemented and evaluated in separate Jupyter notebooks.

### 3.1 XGBoost Classifier
- Core steps:
  - Load and preprocess data.
  - Train an `XGBClassifier` on the training set.
  - Evaluate using:
    - Accuracy, Precision, Recall, F1‑score.
    - Confusion matrix.
    - ROC curve and AUC.
  - Export:
    - Classification report as CSV.
    - Confusion matrix as CSV.
    - ROC points as CSV.
    - Feature importance as CSV (for feature analysis in Tableau).

      <img width="374" height="328" alt="image" src="https://github.com/user-attachments/assets/5100e9ed-132c-4e87-8935-3dbf2bfd89be" />


### 3.2 Support Vector Machine (SVM)
- Notebook: `SVM.ipynb`
- Core steps:
  - Use (usually scaled) features with SVM (e.g. `SVC`).
  - Train on the training data.
  - Evaluate using the same metrics as above.
  - Export classification report, confusion matrix, and ROC data for comparison.

    <img width="384" height="336" alt="image" src="https://github.com/user-attachments/assets/60117830-88bf-4d76-99a4-96dcbd8c6b01" />


### 3.3 Neural Network
- Notebook: `Neural-Network.ipynb`
- Core steps:
  - Define a feed‑forward neural network (e.g. using Keras / sklearn MLP).
  - Train on the training set (with validation).
  - Evaluate performance (Accuracy, Precision, Recall, F1, ROC‑AUC).
  - Export corresponding evaluation CSVs.

 <img width="384" height="335" alt="image" src="https://github.com/user-attachments/assets/d05bcf39-0bc9-45cf-9912-19c54d4b48f5" />

### 3.4 Ensemble Voting Classifier
- Notebook: `Ensemble-Voting-Classifier.ipynb`
- Combines XGBoost, SVM, and Neural Network as base models.
- Uses **hard** or **soft** voting (as configured in the notebook).
- Optionally assigns different **weights** to base models to control their influence.
- Evaluates ensemble performance on the test set and exports:
  - Ensemble classification report.
  - Ensemble confusion matrix.
  - Ensemble ROC data.
  - Misclassification summary per base model (for error analysis in Tableau).
  - Model weights table (for voting weight visualization in Tableau).

<img width="401" height="343" alt="image" src="https://github.com/user-attachments/assets/5625a01b-a0eb-465d-9a0e-d429ec72fd29" />

## 4. Evaluation Metrics

For each model, the following metrics are computed on the test set:

- **Accuracy:** Overall proportion of correctly classified samples.
- **Precision (for High Risk):** Out of patients predicted as high risk, how many actually are high risk.
- **Recall / Sensitivity (for High Risk):** Out of actual high‑risk patients, how many are correctly detected.
- **F1‑score:** Harmonic mean of precision and recall, balancing both.
- **ROC‑AUC:** Measures trade‑off between true positive rate and false positive rate across thresholds.
- **Confusion Matrix:** Counts of True Positives, False Positives, True Negatives, and False Negatives.

Model	Accuracy (%)	WF1 Score	Precision (%)	Recall (%)
XGB	      88         	87	      89	           88
SVM	      89	        90	      91	           89
ANN	      76	        75	      75	           76
EVM     	97	        97	      97	           97


## 5. Visualizations in Tableau

After running the notebooks, several CSV files are generated (metrics, confusion matrices, ROC points, feature importances, misclassification summaries, demographic summaries). These are used to build interactive dashboards in Tableau.

<img width="450" height="419" alt="image" src="https://github.com/user-attachments/assets/4a4940f5-bb6d-43fa-92e1-d764acbda31f" />

### 5.1 Confusion Matrix Heatmap
- Source: confusion matrix CSV for each model.
- In Tableau:
  - Rows: Actual class.
  - Columns: Predicted class.
  - Color/Label: Count of samples.

### 5.2 ROC Curves
- Source: ROC points CSV (FPR, TPR, threshold) for each model.
- In Tableau:
  - Columns: False Positive Rate.
  - Rows: True Positive Rate.
  - Color: Model name (for multiple curves on one plot).

### 5.3 Classification Metrics Bar Chart
- Source: classification report CSV.
- In Tableau:
  - Columns: Metric (Precision, Recall, F1).
  - Rows: Value.
  - Color: Model or Class.

### 5.4 Feature Importance
- Source: feature importance CSV from XGBoost (and optionally coefficients from linear models).
- In Tableau:
  - Rows: Feature.
  - Columns: Importance score.
  - Chart: Horizontal bar chart, sorted descending to highlight top features.

### 5.5 Model Comparison
- Source: a combined CSV summarizing metrics for all models.
- In Tableau:
  - Columns: Model.
  - Rows: Metric value.
  - Color: Metric type (Accuracy, F1, etc.).

### 5.6 Error / Misclassification Analysis
- Source: misclassification summary CSV per model (e.g., `Model, Misclassifications, ErrorRate`).
- In Tableau:
  - Rows: Model.
  - Columns: Misclassifications or ErrorRate.
  - Color: ErrorRate to highlight weaker models.

### 5.7 Data Demographics
- Source: processed dataset including demographics and labels.
- In Tableau:
  - Use fields such as Age group, Gender, Region vs. Risk class.
  - Visualize distribution of high‑risk vs low‑risk across demographic groups.

### 8. Future Enhancements
-Perform systematic hyperparameter tuning (GridSearchCV / RandomizedSearchCV).
-Use cross‑validation for more robust evaluation.
-Add interpretability tools such as SHAP or LIME to explain predictions.
-Deploy the best model as a web API or simple app for clinicians.
-Add automated data validation and logging for real‑world use.

<img width="355" height="591" alt="image" src="https://github.com/user-attachments/assets/89522c1b-68e7-47c6-a44c-79632c105fae" />

<img width="451" height="299" alt="image" src="https://github.com/user-attachments/assets/df4968c1-3564-4db7-922b-ae0c98981004" />

<img width="522" height="367" alt="image" src="https://github.com/user-attachments/assets/0ff14233-c95b-4747-a7b4-00feb3c03612" />

<img width="507" height="346" alt="image" src="https://github.com/user-attachments/assets/4c73a382-4bbb-48c0-9acf-9c467b1ad778" />


<img width="521" height="421" alt="image" src="https://github.com/user-attachments/assets/3ae0e14b-dd32-4296-9375-044abcdc9478" />


