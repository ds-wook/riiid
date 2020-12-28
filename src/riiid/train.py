import pickle
import gc

import pandas as pd
from lightgbm import LGBMClassifier

from data.dataset import train_df
from data.questions import questions_df


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
    "content_correctness": "float16",
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


features = list(features_dict.keys())
target = "answered_correctly"

params = {
    "n_estimators": 5000,
    "num_leaves": 106,
    "max_bin": 169,
    "max_depth": 7,
    "objective": "binary",
    "learning_rate": 0.02,
    "boosting_type": "gbdt",
    "metric": "auc",
}

trains = []
valids = []
num = 5

for i in range(num):
    train_df_clf = train_df.sample(n=1200 * 10000)
    print(f"{i}-fold Sample end")

    gc.collect()
    users = train_df_clf["user_id"].drop_duplicates()

    users = users.sample(frac=0.08)
    users_df = pd.DataFrame()
    users_df["user_id"] = users.values

    valid_df_newuser = pd.merge(
        train_df_clf, users_df, on=["user_id"], how="inner", right_index=True
    )

    gc.collect()
    train_df_clf.drop(valid_df_newuser.index, inplace=True)

    print("Merge train_df_clf, question_df")

    train_df_clf = pd.merge(
        train_df_clf, questions_df, on="content_id", how="left", right_index=True
    )
    valid_df_newuser = pd.merge(
        valid_df_newuser, questions_df, on="content_id", how="left", right_index=True
    )

    valid_df = train_df_clf.sample(frac=0.1)
    train_df_clf.drop(valid_df.index, inplace=True)

    valid_df = valid_df.append(valid_df_newuser)

    gc.collect()

    trains.append(train_df_clf)
    valids.append(valid_df)
    print("train_df_clf length:", len(train_df_clf))
    print("valid_df length：", len(valid_df))

del train_df
del users_df
del users
del valid_df_newuser
del train_df_clf
del valid_df
gc.collect()


for i, (train, valid) in enumerate(zip(trains, valids)):
    X_train = train[features]
    X_valid = valid[features]
    y_train = train[target]
    y_valid = valid[target]

    model = LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_metric="auc",
        verbose=100,
        early_stopping_rounds=50,
    )

    with open(f"../../res/{i}_fold_LGBM.pkl", "wb") as f:
        pickle.dump(model, f)


del X_train
del X_valid
del y_train
del y_valid
del trains
del valids
gc.collect()
