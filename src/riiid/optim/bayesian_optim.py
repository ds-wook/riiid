from typing import Dict, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score

from data.train_dataset import train_dataset


np.seterr(divide='ignore', invalid='ignore')

trains, valids = train_dataset()

features_dict = {
    'timestamp': 'float16',
    'user_interaction_count': 'int16',
    'user_interaction_timestamp_mean': 'float32',
    'lagtime': 'float32',
    'lagtime2': 'float32',
    'lagtime3': 'float32',
    'content_id': 'int16',
    'task_container_id': 'int16',
    'user_lecture_sum': 'int16',
    'user_lecture_lv': 'float16',
    'prior_question_elapsed_time': 'float32',
    'delta_prior_question_elapsed_time': 'int32',
    'user_correctness': 'float16',
    'user_uncorrect_count': 'int16',
    'user_correct_count': 'int16',
    'content_correctness_std': 'float16',
    'content_correct_count': 'int32',
    'content_uncorrect_count': 'int32',
    'content_elapsed_time_mean': 'float16',
    'content_had_explanation_mean': 'float16',
    'content_explation_false_mean': 'float16',
    'content_explation_true_mean': 'float16',
    'task_container_correctness': 'float16',
    'task_container_std': 'float16',
    'task_container_cor_count': 'int32',
    'task_container_uncor_count': 'int32',
    'attempt_no': 'int8',
    'part': 'int8',
    'part_correctness_mean': 'float16',
    'part_correctness_std': 'float16',
    'part_uncor_count': 'int32',
    'part_cor_count': 'int32',
    'tags0': 'int8',
    'tags1': 'int8',
    'tags2': 'int8',
    'tags3': 'int8',
    'tags4': 'int8',
    'tags5': 'int8',
    'part_bundle_id': 'int32',
    'content_sub_bundle': 'int8',
    'prior_question_had_explanation': 'int8',
    'explanation_mean': 'float16',
    'explanation_false_count': 'int16',
    'explanation_true_count': 'int16',
}
features = list(features_dict.keys())
target = 'answered_correctly'

X_train = trains[0][features]
X_valid = valids[0][features]
y_train = trains[0][target]
y_valid = valids[0][target]


def lgb_roc_eval(
        num_leaves: float,
        max_bin: float,
        learning_rate: float) -> float:
    params = {
        'num_leaves': int(round(num_leaves)),
        'max_bin': int(round(max_bin)),
        'learning_rate': max(min(learning_rate, 1), 0)
    }
    lgb_model = lgb.LGBMClassifier(**params)
    lgb_model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric='auc',
                  verbose=100, early_stopping_rounds=50)
    valid_proba = lgb_model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, valid_proba)
    return roc_auc


def lgb_bayes(
        params: Dict[str, Tuple[int, float]]) -> Dict[str, float]:
    lgb_bo = BayesianOptimization(lgb_roc_eval, params)
    lgb_bo.maximize(init_points=5, n_iter=10)
    print(lgb_bo.max['params'])
    return lgb_bo.max['params']
