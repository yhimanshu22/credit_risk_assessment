# 🏦 Credit Risk Assessment in Banking

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Machine%20Learning-Classification-orange.svg" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Status-Completed-success.svg" alt="Status">
</p>

## 📝 Project Overview
This project analyzes credit risk data using machine learning models to classify customers as **"Good"** or **"Bad"** credit risks based on their financial and demographic characteristics. Accurate credit risk assessment is a vital component for banking institutions to minimize losses and optimize lending strategies.

## 📂 Repository Contents

### Source Code (`src/`)
- `config.py`: Centralized configuration for paths and model parameters.
- `data_loader.py`: Functional logic for loading the dataset.
- `preprocessing.py`: Data cleaning, target encoding, and training/test splitting.
- `model.py`: Logistic Regression training and prediction wrappers.
- `evaluation.py`: Performance metrics calculation (Accuracy, TPR, FPR) and reporting.
- `main.py`: Orchestration script to run the full pipeline.

### Data & Documentation
- `data/Credit.csv`: The dataset used for the analysis.
- `Attribute description for Credit data.pdf`: Detailed documentation of the variables.
- `Credit-risk-assessment-in-banking-1.ipynb`: Core Jupyter Notebook for EDA and prototyping.

### Testing
- `tests/`: Directory containing unit tests for core modules.
- `pyproject.toml`: Dependency management configuration.

## 🧠 Methodology & Results
The project implements a **Logistic Regression** baseline. It evaluates the model across different probability thresholds (0.2, 0.35, 0.5) to analyze the trade-off between identifying True Positives (Good risks) and limiting False Positives (Bad risks).

**Best Accuracy (at 0.5 threshold): 78.6%**

## ⚙️ Installation & Execution

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Steps to Run:
1. **Load Environment & Dependencies**:
   ```bash
   uv run python -m src.main
   ```
   *This will automatically create a virtual environment, install dependencies, and run the pipeline.*

2. **Run Tests**:
   ```bash
   uv run python -m pytest tests/
   ```

3. **Check Results**: The pipeline will output performance metrics for each configured threshold directly to your terminal.
