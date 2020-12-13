from typing import Dict, Tuple

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

from cleaning.dataset import X_train, y_train

np.seterr(divide='ignore', invalid='ignore')


def lgb_roc_eval(
        learning_rate: float,
        max_depth: float,
        num_leaves: float,
        colsample_bytree: float,
        subsample: float) -> float:
    params = {
        'n_estimators': 500,
        'learning_rate': max(min(learning_rate, 1), 0),
        'num_leaves': int(round(num_leaves)),
        'max_depth':  int(round(max_depth)),
        'colsample_bytree': max(min(colsample_bytree, 1), 0),
        'subsample': max(min(subsample, 1), 0)
    }
    lgb_model = LGBMClassifier(**params)
    scores = cross_val_score(
                lgb_model, X_train, y_train,
                cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    return roc_auc


def lgb_bayes(
        train: pd.DataFrame,
        target: pd.Series,
        params: Dict[str, Tuple[int, float]]) -> Dict[str, float]:
    lgb_bo = BayesianOptimization(lgb_roc_eval, params)
    lgb_bo.maximize(init_points=5, n_iter=5)
    print(lgb_bo.max['params'])
    return lgb_bo.max['params']
