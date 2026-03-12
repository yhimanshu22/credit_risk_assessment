# Credit-risk-assessment-in-banking
Project Overview
This project analyzes credit risk data using machine learning models to classify customers as "Good" or "Bad" credit risks based on their financial and demographic characteristics. The dataset contains 1000 observations with 62 features, including both numerical and categorical variables.

Dataset
Filename: Credit.csv
Rows: 1000
Columns: 62 (61 features + 1 target variable)
Target Variable: Class (Binary: 0 = "Bad", 1 = "Good")

Variable Types

Numerical Variables (7):
Duration, 
Amount, 
InstallmentRatePercentage, 
ResidenceDuration, 
Age, 
NumberExistingCredits, 
NumberPeopleMaintenance

Categorical Variables (55):
Includes binary encoded features such as:
Telephone,
ForeignWorker,
CheckingAccountStatus categories,
CreditHistory categories,
Purpose categories,
And many others

Data Issues: 
Two categorical columns (Personal.Female.Single and Purpose.Vacation) contain only one class and carry no variance. These should ideally be removed but were retained in this analysis.

Models Implemented:

Logistic Regression:

  Used as a baseline model for binary classification.

  Features were used without one-hot encoding (all variables are numerical or pre-encoded).

  Predictions were made with different probability thresholds (0.2, 0.35, 0.5) to analyze performance metrics.

CatBoost (Suggested):

  Recommended for future work due to the high number of categorical variables.

  CatBoost handles categorical features natively and is well-suited for this type of data.

Key Steps:
Data Loading and Exploration

Loaded the dataset and examined its structure.

Identified numerical and categorical variables.

Encoded the target variable (Class) from text ("Good"/"Bad") to binary (1/0).

Model Training (Logistic Regression)

Split data into features (X) and target (y).

Trained a logistic regression model with high maximum iterations.

Generated predicted probabilities for the positive class.



Performance Evaluation:

Created confusion matrices for different probability thresholds.

Calculated True Positive Rate (TPR), False Positive Rate (FPR), and Accuracy for each threshold.

Generated an ROC curve to visualize model performance.


Results:
Logistic Regression Performance:
Threshold = 0.2:

TPR: 0.994

FPR: 0.867

Accuracy: 0.736

Threshold = 0.35:

TPR: 0.967

FPR: 0.683

Accuracy: 0.772

Threshold = 0.5:

TPR: 0.899

FPR: 0.477

Accuracy: 0.786

Dependencies: 
Python 3.x 
Libraries:
numpy,

pandas,

matplotlib,

seaborn,

scikit-learn,
