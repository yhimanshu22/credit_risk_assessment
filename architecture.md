# Credit Risk Assessment — Architecture

## Overview

Binary classification pipeline that predicts whether a loan applicant is a **Good** or **Bad** credit risk. The production flow lives in `src/` and is orchestrated by `main.py`. Configuration is centralized in `config.py`.

| Item | Value |
|------|-------|
| Dataset | `data/Credit.csv` — 1,000 rows, 62 columns |
| Features | 61 (7 numerical + 2 binary + 52 one-hot categoricals) |
| Target | `Class` — Good (1) / Bad (0) |
| Models | Logistic Regression, Decision Tree, Bagging, Random Forest, SVM (RBF) |
| Tuning | GridSearchCV (5-fold CV) for tree-based ensemble models |
| Split | 70% train / 30% test (`random_state=42`) |

---

## ML Pipeline

```mermaid
flowchart TB
    START([uv run python -m src.main]) --> MAIN[main.py · run_pipeline]

    MAIN --> S1

    subgraph S1["① Load Data — data_loader.py"]
        L1[Read Credit.csv via pandas]
        L2[Handle FileNotFoundError]
        L1 --> L2
    end

    subgraph S2["② Preprocess — preprocessing.py"]
        P1[Map Class: Good → 1, Bad → 0]
        P2[Handle whitespace variants]
        P3[Move Class to last column]
        P4[Cast Class to int]
        P1 --> P2 --> P3 --> P4
    end

    subgraph S3["③ Split — preprocessing.py"]
        SP1[X = 61 features · y = Class]
        SP2[train_test_split · test_size=0.3]
        SP3["Train: 700 samples · Test: 300 samples"]
        SP1 --> SP2 --> SP3
    end

    subgraph S4A["④a Logistic Regression — model.py"]
        LR1[LogisticRegression max_iter=1e8]
        LR2[fit on X_train]
        LR3[predict_proba on X_test]
        LR1 --> LR2 --> LR3
    end

    subgraph S4B["④b Other Models — model.py"]
        M1[Decision Tree · Bagging · Random Forest]
        M2[GridSearchCV · cv=5 · scoring=accuracy]
        M3[Return best_estimator_]
        M4[SVM RBF · probability=True]
        M1 --> M2 --> M3
        M4
    end

    subgraph S5["⑤ Evaluate — evaluation.py"]
        E1["LR: thresholds 0.2, 0.35, 0.5"]
        E2["Others: predict + predict_proba"]
        E3[Confusion matrix]
        E4[Accuracy · Precision · Recall · TPR · FPR · FNR · AUC]
        E5[Per-model summary + comparison table]
        E1 --> E3
        E2 --> E3
        E3 --> E4 --> E5
    end

    END([Pipeline complete])

    S1 --> S2 --> S3 --> S4A --> S5
    S3 --> S4B --> S5 --> END
```

---

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        CSV["Credit.csv<br/>1000 × 62"]
        NUM["Numerical (7)<br/>Duration, Amount, Age, ..."]
        BIN["Binary (2)<br/>Telephone, ForeignWorker"]
        CAT["One-hot (52)<br/>CreditHistory, Purpose, Job, ..."]
        TGT["Class: Good / Bad"]
    end

    subgraph Processed
        DF["DataFrame<br/>1000 × 62<br/>Class = 0 or 1"]
    end

    subgraph Split
        TR["X_train · y_train<br/>700 × 61"]
        TE["X_test · y_test<br/>300 × 61"]
    end

    subgraph Output
        PROB["Probabilities<br/>P(Good) × 300"]
        MET["Metrics per model<br/>Accuracy · Precision · Recall · AUC · CM"]
        CMP["Model comparison table"]
    end

    CSV --> NUM
    CSV --> BIN
    CSV --> CAT
    CSV --> TGT
    NUM --> DF
    BIN --> DF
    CAT --> DF
    TGT --> DF
    DF --> TR
    DF --> TE
    TR --> PROB
    TE --> PROB
    PROB --> MET
    MET --> CMP
```

---

## Module Structure

```mermaid
graph TD
    MAIN[main.py] --> CONFIG[config.py]
    MAIN --> DL[data_loader.py]
    MAIN --> PP[preprocessing.py]
    MAIN --> MD[model.py]
    MAIN --> EV[evaluation.py]

    CONFIG -.->|paths & params| DL
    CONFIG -.->|TEST_SIZE, RANDOM_STATE| PP
    CONFIG -.->|MAX_ITER, param grids| MD
    CONFIG -.->|THRESHOLDS, CV_FOLDS| EV

    DL -->|DataFrame| PP
    PP -->|X_train, X_test, y_train, y_test| MD
    MD -->|models, y_probs, y_pred| EV
```

| Module | Key Functions | Responsibility |
|--------|---------------|----------------|
| `config.py` | — | Paths, split params, thresholds, GridSearchCV grids |
| `data_loader.py` | `load_credit_data()` | Load CSV from `data/Credit.csv` |
| `preprocessing.py` | `preprocess_data()`, `split_credit_data()` | Target encoding, column order, train/test split |
| `model.py` | `train_*()`, `get_all_trainers()`, `get_prediction_probabilities()` | Train all models; tune tree ensembles via GridSearchCV |
| `evaluation.py` | `calculate_metrics()`, `calculate_predict_metrics()`, `print_model_comparison()` | Threshold and hard-prediction metrics, comparison table |
| `main.py` | `run_pipeline()` | Orchestrates load → preprocess → train → evaluate |

---

## Models

| Model | Training | Threshold tuning |
|-------|----------|------------------|
| **Logistic Regression** | Direct fit (`max_iter=1e8`) | Yes — evaluated at 0.2, 0.35, 0.5 |
| **Decision Tree** | GridSearchCV on `max_depth` | No — uses `predict()` |
| **Bagging** | GridSearchCV on `n_estimators`, `max_features` | No |
| **Random Forest** | GridSearchCV on `n_estimators`, `max_features` | No |
| **SVM (RBF)** | Direct fit (`probability=True`) | No |

### GridSearchCV configuration

Defined in `config.py`:

| Model | Parameter grid |
|-------|----------------|
| Decision Tree | `max_depth`: [3, 4, 5, 6, 7, 8, 10, 20] |
| Bagging | `n_estimators`: [100, 150, 200], `max_features`: [0.5, 0.7, 1.0] |
| Random Forest | `n_estimators`: [100, 150, 200], `max_features`: [0.5, 0.7, 1.0] |

All grid searches use **5-fold cross-validation** with **accuracy** as the scoring metric. The pipeline prints the best parameters and CV score, then retrains using `best_estimator_`.

---

## Threshold Evaluation Logic (Logistic Regression)

```mermaid
flowchart TD
    PROB["Model output: P(Good)"] --> TH{prob > threshold?}
    TH -->|Yes| GOOD["Predict: Good (1)"]
    TH -->|No| BAD["Predict: Bad (0)"]

    GOOD --> CM[Confusion Matrix]
    BAD --> CM

    CM --> ACC["Accuracy"]
    CM --> PREC["Precision"]
    CM --> REC["Recall / TPR"]
    CM --> FPR["FPR"]
    CM --> FNR["FNR"]
    PROB --> AUC["AUC (threshold-independent)"]
```

| Threshold | Trade-off |
|-----------|-----------|
| **0.2** | High TPR (~99%) — approve almost all Good, but also many Bad |
| **0.35** | Balanced recall — ~95% Good approved |
| **0.5** | Default — best accuracy (~77%), moderate FPR |

---

## Project Layout

```
credit_risk_assessment/
├── data/Credit.csv              # Dataset
├── src/
│   ├── config.py                # Paths, thresholds, GridSearchCV grids
│   ├── data_loader.py           # Step 1: Load
│   ├── preprocessing.py         # Step 2–3: Preprocess & split
│   ├── model.py                 # Step 4: Train & tune models
│   ├── evaluation.py            # Step 5: Metrics & comparison
│   └── main.py                  # Pipeline entry point
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_evaluation.py
│   └── test_model_tuning.py
├── Credit-risk-assessment-in-banking-1.ipynb   # EDA, plots, KNN experiments
├── pyproject.toml               # Dependencies (uv)
└── architecture.md              # This file
```

---

## Run

```bash
uv run python -m src.main
uv run python -m pytest tests/
```

The pipeline prints:
1. Logistic regression metrics at each threshold (0.2, 0.35, 0.5)
2. GridSearchCV best params for Decision Tree, Bagging, and Random Forest
3. Test-set metrics for all non-LR models
4. A side-by-side model comparison table
