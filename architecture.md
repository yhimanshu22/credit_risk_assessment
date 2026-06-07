# Credit Risk Assessment — Architecture

## Overview

Binary classification pipeline that predicts whether a loan applicant is a **Good** or **Bad** credit risk. The production flow lives in `src/` and is orchestrated by `main.py`. Configuration is centralized in `config.py`.

| Item | Value |
|------|-------|
| Dataset | `data/Credit.csv` — 1,000 rows, 62 columns |
| Features | 61 (7 numerical + 54 one-hot encoded categoricals) |
| Target | `Class` — Good (1) / Bad (0) |
| Model | Logistic Regression |
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

    subgraph S4["④ Train — model.py"]
        T1[LogisticRegression max_iter=1e8]
        T2[model.fit X_train, y_train]
        T1 --> T2
    end

    subgraph S5["⑤ Predict — model.py"]
        PR1[predict_proba X_test]
        PR2["Output: P(Good) per sample"]
        PR1 --> PR2
    end

    subgraph S6["⑥ Evaluate — evaluation.py"]
        E1["For each threshold: 0.2, 0.35, 0.5"]
        E2["y_pred = 1 if prob > threshold else 0"]
        E3[Build confusion matrix]
        E4[Compute Accuracy · TPR · FPR]
        E5[Print performance summary]
        E1 --> E2 --> E3 --> E4 --> E5
    end

    END([Pipeline complete])

    S1 --> S2 --> S3 --> S4 --> S5 --> S6 --> END
```

---

## Data Flow

```mermaid
flowchart LR
    subgraph Input
        CSV["Credit.csv<br/>1000 × 62"]
        NUM["Numerical (7)<br/>Duration, Amount, Age, ..."]
        CAT["One-hot (54)<br/>CreditHistory, Purpose, Job, ..."]
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
        MET["Metrics × 3 thresholds<br/>Accuracy · TPR · FPR · CM"]
    end

    CSV --> NUM
    CSV --> CAT
    CSV --> TGT
    NUM --> DF
    CAT --> DF
    TGT --> DF
    DF --> TR
    DF --> TE
    TR --> PROB
    TE --> PROB
    PROB --> MET
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
    CONFIG -.->|MAX_ITER| MD
    CONFIG -.->|THRESHOLDS| EV

    DL -->|DataFrame| PP
    PP -->|X_train, X_test, y_train, y_test| MD
    MD -->|y_probs| EV
```

| Module | Key Functions | Responsibility |
|--------|---------------|----------------|
| `config.py` | — | Paths, `RANDOM_STATE=42`, `TEST_SIZE=0.3`, `THRESHOLDS`, `MAX_ITER` |
| `data_loader.py` | `load_credit_data()` | Load CSV from `data/Credit.csv` |
| `preprocessing.py` | `preprocess_data()`, `split_credit_data()` | Target encoding, column order, train/test split |
| `model.py` | `train_logistic_regression()`, `get_prediction_probabilities()` | Fit LR model, return P(Good) |
| `evaluation.py` | `calculate_metrics()`, `print_performance_summary()` | Threshold-based metrics and reporting |
| `main.py` | `run_pipeline()` | Wires all steps in sequence |

---

## Threshold Evaluation Logic

```mermaid
flowchart TD
    PROB["Model output: P(Good)"] --> TH{prob > threshold?}
    TH -->|Yes| GOOD["Predict: Good (1)"]
    TH -->|No| BAD["Predict: Bad (0)"]

    GOOD --> CM[Confusion Matrix]
    BAD --> CM

    CM --> TPR["TPR = TP / (TP + FN)<br/>% of Good correctly approved"]
    CM --> FPR["FPR = FP / (FP + TN)<br/>% of Bad wrongly approved"]
    CM --> ACC["Accuracy = (TP + TN) / Total"]
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
│   ├── config.py                # Configuration
│   ├── data_loader.py           # Step 1: Load
│   ├── preprocessing.py         # Step 2–3: Preprocess & Split
│   ├── model.py                 # Step 4–5: Train & Predict
│   ├── evaluation.py            # Step 6: Evaluate
│   └── main.py                  # Pipeline entry point
├── tests/                       # Unit tests (loader, preprocessing)
├── Credit-risk-assessment-in-banking-1.ipynb   # EDA & model experiments
├── pyproject.toml               # Dependencies (uv)
└── architecture.md              # This file
```

---

## Run

```bash
uv run python -m src.main
uv run python -m pytest tests/
```
