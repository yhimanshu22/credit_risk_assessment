import pandas as pd
import pytest

from src.config import (
    BAGGING_PARAM_GRID,
    CV_FOLDS,
    DECISION_TREE_PARAM_GRID,
    RANDOM_FOREST_PARAM_GRID,
)
from src.model import train_decision_tree


def test_param_grids_match_notebook_setup():
    assert CV_FOLDS == 5
    assert DECISION_TREE_PARAM_GRID["max_depth"] == [3, 4, 5, 6, 7, 8, 10, 20]
    assert BAGGING_PARAM_GRID["n_estimators"] == [100, 150, 200]
    assert RANDOM_FOREST_PARAM_GRID["max_features"] == [0.5, 0.7, 1.0]


def test_train_decision_tree_returns_fitted_model():
    df = pd.read_csv("data/Credit.csv")
    df["Class"] = df["Class"].str.strip().map({"Good": 1, "Bad": 0})
    X = df.drop("Class", axis=1).head(200)
    y = df["Class"].head(200)

    model = train_decision_tree(X, y)

    assert hasattr(model, "predict")
    assert model.max_depth is not None
