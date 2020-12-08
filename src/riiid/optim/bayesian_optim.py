from typing import Dict, Tuple
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from cleaning.dataset import X_train, X_valid, y_train, y_valid
import numpy as np
import pandas as pd
np.seterr(divide='ignore', invalid='ignore')


def lgb_roc_eval(
        learning_rate: float,
        max_depth: float,
        num_leaves: float,
        feature_fraction: float,
        subsample: float) -> float:
    params = {
        'n_estimators': 500,
        'learning_rate': max(min(learning_rate, 1), 0),
        'num_leaves': int(round(num_leaves)),
        'max_depth':  int(round(max_depth)),
        'feature_fraction': max(min(feature_fraction, 1), 0),
        'subsample': max(min(subsample, 1), 0)
    }
    lgb_model = LGBMClassifier(**params)
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric='auc',
        verbose=100,
        early_stopping_rounds=100)
    valid_proba = lgb_model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, valid_proba)
    return roc_auc


def lgb_bayes(
        train: pd.DataFrame,
        target: pd.Series,
        params: Dict[str, Tuple[int, float]]) -> Dict[str, float]:
    lgb_bo = BayesianOptimization(lgb_roc_eval, params)
    lgb_bo.maximize(init_points=5, n_iter=25)
    print(lgb_bo.max['params'])
    return lgb_bo.max['params']
