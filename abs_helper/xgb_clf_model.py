# abs_helper/modeling/xgb.py
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV

def build_xgb_pipeline(cat_cols, num_cols):
    """Return a preprocessing + XGB pipeline."""
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
    """Generate parameter grid for XGB, including scale_pos_weight."""
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
    """Run RandomizedSearchCV with TimeSeriesSplit and return the fitted search object."""
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
    # split off a small calibration slice from the end of TRAIN
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
        Calibrate a probabilistic model with isotonic (default) or sigmoid scaling.
        Uses out-of-fold calibration if groups are supplied.
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