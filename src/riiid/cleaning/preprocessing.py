import datatable as dt
import pandas as pd
import numpy as np
from typing import Tuple
np.seterr(divide='ignore', invalid='ignore')
path = '../../kaggle/input/riiid-test-answer-prediction/'


def data_preprocessing(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = 'answered_correctly'
    columns = ['user_id', 'content_id', 'prior_question_elapsed_time',
               'prior_question_had_explanation']
    train = dt.fread(path + 'train.csv')
    train = train.to_pandas()

    train =\
        train.loc[(train['content_type_id'] == False), columns + [target_col]]

    user = train.groupby('user_id')[target_col].agg(['sum', 'count'])
    user['user_percent_correct'] = user['sum'] / user['count']
    user.drop(['sum'], axis=1, inplace=True)

    question = train.groupby('content_id')[target_col].agg(['sum', 'count'])
    question['question_precent_correct'] = question['sum'] / question['count']
    question.drop(['sum', 'count'], axis=1, inplace=True)

    train = train.join(user, on='user_id')
    train = train.join(question, on='content_id')
    train.reset_index(drop=True, inplace=True)
    train['prior_question_had_explanation'] =\
        train['prior_question_had_explanation'].fillna(False).astype(np.int8)
    numeric_features = ['prior_question_elapsed_time', 'count',
                        'user_percent_correct', 'question_precent_correct']
    categorical_features = ['prior_question_had_explanation']
    target = train[target_col]

    return train[numeric_features + categorical_features], target
