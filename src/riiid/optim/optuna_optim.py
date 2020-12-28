import gc

from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
import numpy as np

from data.train_dataset import train_dataset

_ = np.seterr(divide="ignore", invalid="ignore")

trains, valids = train_dataset()

features_dict = {
    "timestamp": "float16",
    "user_interaction_count": "int16",
    "user_interaction_timestamp_mean": "float32",
    "lagtime": "float32",
    "lagtime2": "float32",
    "lagtime3": "float32",
    "content_id": "int16",
    "task_container_id": "int16",
    "user_lecture_sum": "int16",
    "user_lecture_lv": "float16",
    "prior_question_elapsed_time": "float32",
    "delta_prior_question_elapsed_time": "int32",
    "user_correctness": "float16",
    "user_uncorrect_count": "int16",
    "user_correct_count": "int16",
    "content_correctness_std": "float16",
    "content_correct_count": "int32",
    "content_uncorrect_count": "int32",
    "content_elapsed_time_mean": "float16",
    "content_had_explanation_mean": "float16",
    "content_explation_false_mean": "float16",
    "content_explation_true_mean": "float16",
    "task_container_correctness": "float16",
    "task_container_std": "float16",
    "task_container_cor_count": "int32",
    "task_container_uncor_count": "int32",
    "attempt_no": "int8",
    "part": "int8",
    "part_correctness_mean": "float16",
    "part_correctness_std": "float16",
    "part_uncor_count": "int32",
    "part_cor_count": "int32",
    "tags0": "int8",
    "tags1": "int8",
    "tags2": "int8",
    "tags3": "int8",
    "tags4": "int8",
    "tags5": "int8",
    "part_bundle_id": "int32",
    "content_sub_bundle": "int8",
    "prior_question_had_explanation": "int8",
    "explanation_mean": "float16",
    "explanation_false_count": "int16",
    "explanation_true_count": "int16",
}

categorical_columns = [
    "content_id",
    "task_container_id",
    "part",
    "tags1",
    "tags2",
    "tags3",
    "tags4",
    "tags5",
    "part_bundle_id",
    "content_sub_bundle",
    "prior_question_had_explanation",
]


features = list(features_dict.keys())
target = "answered_correctly"
X_train = trains[0][features]
X_valid = valids[0][features]
y_train = trains[0][target]
y_valid = valids[0][target]

gc.collect()


def objective(trial):
    params = {
        "n_estimators": 5000,
        "num_leaves": trial.suggest_int("num_leaves", 10, 400),
        "max_bin": trial.suggest_int("max_bin", 10, 400),
        "objective": "binary",
        "learning_rate": 0.02,
        "boosting_type": "gbdt",
        "metric": "auc",
    }

    lgb_model = LGBMClassifier(**params)
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="auc",
        verbose=100,
        early_stopping_rounds=50,
    )
    valid_proba = lgb_model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, valid_proba)

    return roc_auc
