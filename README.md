# 🏦 Credit Risk Assessment in Banking

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Machine%20Learning-Classification-orange.svg" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status">
</p>

## 📝 Project Overview
This project analyzes credit risk data using machine learning models to classify customers as **"Good"** or **"Bad"** credit risks based on their financial and demographic characteristics. Accurate credit risk assessment is a vital component for banking institutions to minimize losses and optimize lending strategies.

## 📂 Repository Contents
- `Credit-risk-assessment-in-banking-1.ipynb`: Core Jupyter Notebook containing Exploratory Data Analysis (EDA), data preprocessing, and model training/evaluation.
- `Credit.csv`: The dataset used for the analysis.
- `Attribute description for Credit data.pdf`: Detailed documentation of the variables and attributes present in the dataset.
- `README.md`: Project documentation.

## 📊 Dataset Information
The dataset consists of **1000 observations** and **62 features** (61 predictor variables + 1 target variable). 

- **Target Variable (`Class`)**: Binary classification identifying the borrower's risk type.
  - `0` = "Bad" Credit Risk
  - `1` = "Good" Credit Risk

### Feature Breakdown
* **Numerical Variables (7):** `Duration`, `Amount`, `InstallmentRatePercentage`, `ResidenceDuration`, `Age`, `NumberExistingCredits`, `NumberPeopleMaintenance`
* **Categorical Variables (55):** Contains binary encoded features. Examples include `Telephone`, `ForeignWorker`, `CheckingAccountStatus` categories, `CreditHistory` categories, `Purpose` categories, and more.

> **💡 Note on Data Variance:** Two categorical columns (`Personal.Female.Single` and `Purpose.Vacation`) contain only a single class across the entire dataset, meaning they provide no variance. They were retained in this baseline analysis but are recommended for removal in further optimization.

## 🧠 Methodology & Machine Learning Models

### 1. Data Processing
- Loaded and explored the dataset to map out its structure and feature types.
- Numerically encoded the target variable (`Class`) from text labels ("Good"/"Bad") into a binary format (`1`/`0`).

### 2. Logistic Regression (Baseline Model)
- Used as the primary baseline model for binary classification.
- Features were ingested naturally without additional one-hot encoding (as variables were mostly numerical or pre-encoded).
- Trained using a high number of maximum iterations for optimal convergence.
- Predicted probabilities were explored across various classification thresholds (0.2, 0.35, 0.5) to analyze the trade-off between identifying True Positives and avoiding False Positives.

### 3. CatBoost (Suggested for Future Work)
- Due to the high dimensionality of categorical variables natively present in the data, **CatBoost** is recommended for subsequent model iterations. CatBoost handles categorical features natively and is highly robust for this specific predictive structure.

## 📈 Results & Performance Evaluation

The model was rigorously evaluated using Confusion Matrices, True Positive Rate (TPR), False Positive Rate (FPR), Accuracy, and an ROC Curve.

Below is the performance summary of the **Logistic Regression** model across different probability thresholds:

| Probability Threshold | True Positive Rate (TPR) | False Positive Rate (FPR) | Accuracy |
| :---: | :---: | :---: | :---: |
| **0.20** | 0.994 | 0.867 | 73.6% |
| **0.35** | 0.967 | 0.683 | 77.2% |
| **0.50** | 0.899 | 0.477 | **78.6%** |

*Adjusting the threshold to 0.50 provided the highest overall model accuracy, substantially reducing the False Positive Rate when compared with lower thresholds.*

## ⚙️ Installation & Dependencies

To setup and run this project locally, ensure you have Python 3.x installed along with the required libraries.

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Steps to Run:
1. Clone this repository or download the files.
2. Ensure `Credit.csv` is in the same directory as the Jupyter Notebook.
3. Open `Credit-risk-assessment-in-banking-1.ipynb` using Jupyter Notebook, JupyterLab, or VS Code.
4. Run the cells sequentially to reproduce the data analysis, visualizations, and model results.
