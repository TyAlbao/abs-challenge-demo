"""
xgb_clf_model.py
-----------------
Handles XGBoost model construction, training, and calibration for the ABS Challenge system.

Responsibilities:
- Building preprocessing + XGB pipelines
- Defining hyperparameter search distributions
- Running RandomizedSearchCV with time-series or group splits
- Calibrating probabilities with isotonic or sigmoid methods
"""

# Third-party
import numpy as np
from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    RandomizedSearchCV,
    TimeSeriesSplit,
    GroupKFold
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.calibration import CalibratedClassifierCV


def build_xgb_pipeline(cat_cols, num_cols):
    """
    Construct preprocessing + XGBoost pipeline.

    - One-hot encodes categorical features
    - Passes through numeric features
    - Adds XGB classifier with reasonable defaults
    """
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    pipe = Pipeline([
        ("prep", preproc),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",    # "gpu_hist" if GPU
            random_state=42,
            n_jobs=-1,
        ))
    ])
    return pipe

def get_xgb_param_dist(y_train):
    """
    Generate hyperparameter distribution for RandomizedSearchCV.

    Includes adjustments for class imbalance via scale_pos_weight.
    """
    pos_ratio = y_train.mean()
    neg_ratio = 1 - pos_ratio
    spw = neg_ratio / pos_ratio

    return {
        "clf__n_estimators": [200, 300, 400, 600, 800],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
        "clf__subsample": [0.7, 0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "clf__reg_lambda": [0.5, 1.0, 2.0, 5.0, 10.0],
        "clf__reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "clf__scale_pos_weight": [1.0, spw * 0.5, spw, spw * 1.5, spw * 2.0]
    }

def train_xgb(X_train, y_train, cat_cols, num_cols, n_iter=40, scoring="roc_auc"):
    """
    Train an XGBClassifier with hyperparameter search.

    Args:
        X_train, y_train: Training data.
        cat_cols, num_cols: Feature splits.
        n_iter: Number of search iterations.
        scoring: Metric for evaluation (default: AUC).
    Returns:
        Fitted RandomizedSearchCV object.
    """
    pipe = build_xgb_pipeline(cat_cols, num_cols)
    param_dist = get_xgb_param_dist(y_train)

    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring if isinstance(scoring, str) else {"roc_auc": "roc_auc"},
        refit="roc_auc" if not isinstance(scoring, str) else scoring,
        cv=tscv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
    )
    search.fit(X_train, y_train)
    return search

def calibrate_probs(model, X_train, y_train):
    """
    Fit isotonic calibration on held-out slice of training data.

    Returns calibrated model for improved probability estimates.
    """
    cut_cal = int(len(X_train) * 0.9)
    X_core, y_core = X_train.iloc[:cut_cal], y_train[:cut_cal]
    X_cal,  y_cal  = X_train.iloc[cut_cal:], y_train[cut_cal:]

    best_pipe = clone(model)
    best_pipe.fit(X_core, y_core)

    calib_iso = CalibratedClassifierCV(best_pipe, method="isotonic")
    calib_iso.fit(X_cal, y_cal)
    return calib_iso

def calibrate_model(model, X, y, groups=None, method="isotonic", cv_splits=5):
    """
    Perform out-of-fold calibration (isotonic or sigmoid).

    Args:
        model: Base model.
        X, y: Data.
        groups (optional): Group splits for CV.
        method: Calibration method.
        cv_splits: Number of folds.
    Returns:
        Calibrated model.
    """
    if groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
    else:
        cv = cv_splits  # e.g., 5 folds

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=cv
    )
    calibrated.fit(X, y)
    return calibrated