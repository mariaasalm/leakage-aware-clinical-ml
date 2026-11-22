"""Evaluation strategies: naive (leaky) vs grouped vs nested-grouped cross-validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold, StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURES


def build_pipeline(model: str = "rf", seed: int = 0) -> Pipeline:
    """Scaling lives *inside* the pipeline so it is re-fitted on each training fold only."""
    if model == "logreg":
        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=seed, n_jobs=1)
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def _scores(y_true, y_prob) -> dict:
    return {"auroc": roc_auc_score(y_true, y_prob), "auprc": average_precision_score(y_true, y_prob)}


def cross_validate(df: pd.DataFrame, strategy: str, model: str = "rf",
                   n_splits: int = 5, seed: int = 0) -> dict:
    """Out-of-fold AUROC/AUPRC under a given splitting strategy.

    strategy:
      - "row"     : StratifiedKFold over records (LEAKY: same patient in train and test)
      - "patient" : StratifiedGroupKFold over patients (correct)
    """
    X, y, groups = df[FEATURES].to_numpy(), df["label"].to_numpy(), df["patient_id"].to_numpy()
    if strategy == "row":
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    elif strategy == "patient":
        splits = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, groups)
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    oof = np.zeros(len(y))
    for tr, te in splits:
        pipe = build_pipeline(model, seed).fit(X[tr], y[tr])
        oof[te] = pipe.predict_proba(X[te])[:, 1]
    return _scores(y, oof)


def nested_cross_validate(df: pd.DataFrame, n_outer: int = 5, n_inner: int = 3, seed: int = 0) -> dict:
    """Hyper-parameter search in an inner grouped loop; unbiased estimate from the outer loop."""
    X, y, groups = df[FEATURES].to_numpy(), df["label"].to_numpy(), df["patient_id"].to_numpy()
    grid = {"clf__max_depth": [3, 6, None], "clf__min_samples_leaf": [1, 5]}
    outer = StratifiedGroupKFold(n_outer, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in outer.split(X, y, groups):
        search = GridSearchCV(build_pipeline("rf", seed), grid, scoring="roc_auc",
                              cv=GroupKFold(n_inner), n_jobs=1)
        search.fit(X[tr], y[tr], groups=groups[tr])
        oof[te] = search.predict_proba(X[te])[:, 1]
    return _scores(y, oof)
