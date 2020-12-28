import gc
from typing import List, Tuple

import pandas as pd

from data.questions import questions_df
from data.dataset import train_df


def train_dataset() -> Tuple[List, List]:
    trains = []
    valids = []
    num = 1

    for i in range(num):
        train_df_clf = train_df.sample(n=1200 * 10000)
        print("sample end")

        users = train_df_clf["user_id"].drop_duplicates()

        users = users.sample(frac=0.08)
        users_df = pd.DataFrame()
        users_df["user_id"] = users.values

        valid_df_newuser = pd.merge(
            train_df_clf, users_df, on=["user_id"], how="inner", right_index=True
        )
        del users_df
        gc.collect()
        train_df_clf.drop(valid_df_newuser.index, inplace=True)

        print("train_df_clf, question_df merge")

        train_df_clf = pd.merge(
            train_df_clf, questions_df, on="content_id", how="left", right_index=True
        )

        valid_df_newuser = pd.merge(
            valid_df_newuser,
            questions_df,
            on="content_id",
            how="left",
            right_index=True,
        )

        valid_df = train_df_clf.sample(frac=0.1)
        train_df_clf.drop(valid_df.index, inplace=True)

        valid_df = valid_df.append(valid_df_newuser)
        del users
        del valid_df_newuser
        gc.collect()

        trains.append(train_df_clf)
        valids.append(valid_df)
        print("train_df_clf length:", len(train_df_clf))
        print("valid_df length：", len(valid_df))

    return trains, valids
