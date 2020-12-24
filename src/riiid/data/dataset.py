import gc

import datatable as dt
import pandas as pd
import numpy as np

np.seterr(divide='ignore', invalid='ignore')
path = '../../input/riiid-test-answer-prediction/'


def train_preprocessing(path: str) -> pd.DataFrame:
    data_types_dict = {
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'task_container_id': 'int16',
        'answered_correctly': 'int8',
        'prior_question_elapsed_time': 'float32',
        'prior_question_had_explanation': 'bool'
    }
    target = 'answered_correctly'
    train_df = dt.fread(path + 'train.csv',
                        columns=set(data_types_dict.keys())).to_pandas()
    lectures_df = pd.read_csv(path + 'lectures.csv')
    lectures_df['type_of'] =\
        lectures_df['type_of'].replace('solving question', 'solving_question')
    lectures_df =\
        pd.get_dummies(lectures_df, columns=['part', 'type_of'])
    part_lectures_columns =\
        [col for col in lectures_df.columns if col.startswith('part')]
    types_of_lectures_columns =\
        [col for col in lectures_df.columns if col.startswith('type_of_')]
    train_lectures = train_df[train_df.content_type_id == 1]\
        .merge(lectures_df, left_on='content_id',
               right_on='lecture_id', how='left')
    user_lecture_stats_part = train_lectures\
        .groupby('user_id', as_index=False)[
            part_lectures_columns + types_of_lectures_columns].sum()
    lecturedata_types_dict = {
        'user_id': 'int32',
        'part_1': 'int8',
        'part_2': 'int8',
        'part_3': 'int8',
        'part_4': 'int8',
        'part_5': 'int8',
        'part_6': 'int8',
        'part_7': 'int8',
        'type_of_concept': 'int8',
        'type_of_intention': 'int8',
        'type_of_solving_question': 'int8',
        'type_of_starter': 'int8'
    }
    user_lecture_stats_part =\
        user_lecture_stats_part.astype(lecturedata_types_dict)

    for column in user_lecture_stats_part.columns:
        if(column != 'user_id'):
            user_lecture_stats_part[column] =\
                (user_lecture_stats_part[column] > 0).astype('int8')

    # clearing memory
    del(train_lectures)
    gc.collect()

    user_lecture_agg =\
        train_df.groupby('user_id')['content_type_id'].agg(['sum', 'count'])
    user_lecture_agg = user_lecture_agg.astype('int16')

    cum = train_df.groupby('user_id')['content_type_id']\
        .agg(['cumsum', 'cumcount'])
    cum['cumcount'] += 1
    train_df['user_interaction_count'] = cum['cumcount']
    train_df['user_interaction_timestamp_mean'] =\
        train_df['timestamp'] / cum['cumcount']

    train_df['user_lecture_sum'] = cum['cumsum']
    train_df['user_lecture_lv'] = cum['cumsum'] / cum['cumcount']

    train_df.user_lecture_lv = train_df.user_lecture_lv.astype('float16')
    train_df.user_lecture_sum = train_df.user_lecture_sum.astype('int8')

    train_df.user_interaction_count =\
        train_df.user_interaction_count.astype('int16')
    train_df['user_interaction_timestamp_mean'] =\
        train_df['user_interaction_timestamp_mean'] / (1000 * 3600)
    train_df.user_interaction_timestamp_mean =\
        train_df.user_interaction_timestamp_mean.astype('float32')

    del cum
    gc.collect()
    train_df['prior_question_had_explanation'].fillna(False, inplace=True)
    train_df = train_df.astype(data_types_dict)
    train_df = train_df[train_df[target] != -1].reset_index(drop=True)

    content_explation_agg = train_df[
        ['content_id', 'prior_question_had_explanation', target]]\
        .groupby(['content_id', 'prior_question_had_explanation'])[target]\
        .agg(['mean'])

    content_explation_agg = content_explation_agg.unstack()
    content_explation_agg = content_explation_agg.reset_index()
    content_explation_agg.columns =\
        ['content_id', 'content_explation_false_mean',
         'content_explation_true_mean']

    content_explation_agg.content_id =\
        content_explation_agg.content_id.astype('int16')
    content_explation_agg.content_explation_false_mean =\
        content_explation_agg.content_explation_false_mean.astype('float16')
    content_explation_agg.content_explation_true_mean =\
        content_explation_agg.content_explation_true_mean.astype('float16')

    train_df["attempt_no"] = 1
    train_df.attempt_no = train_df.attempt_no.astype('int8')
    attempt_no_agg = train_df.groupby(["user_id", "content_id"])['attempt_no']\
        .agg(['sum']).astype('int8')
    train_df["attempt_no"] = train_df[["user_id", "content_id", 'attempt_no']]\
        .groupby(["user_id", "content_id"])["attempt_no"].cumsum()

    attempt_no_agg = attempt_no_agg[attempt_no_agg['sum'] > 1]
    prior_question_elapsed_time_mean =\
        train_df['prior_question_elapsed_time'].mean()
    train_df['prior_question_elapsed_time']\
        .fillna(prior_question_elapsed_time_mean, inplace=True)

    max_timestamp_u = train_df[['user_id', 'timestamp']]\
        .groupby(['user_id']).agg(['max']).reset_index()
    max_timestamp_u.columns = ['user_id', 'max_time_stamp']
    max_timestamp_u.user_id = max_timestamp_u.user_id.astype('int32')

    train_df['lagtime'] = train_df.groupby('user_id')['timestamp'].shift()
    max_timestamp_u2 = train_df[['user_id', 'lagtime']].groupby(['user_id'])\
        .agg(['max']).reset_index()
    max_timestamp_u2.columns = ['user_id', 'max_time_stamp2']
    max_timestamp_u2.user_id = max_timestamp_u2.user_id.astype('int32')

    train_df['lagtime'] = train_df['timestamp'] - train_df['lagtime']
    lagtime_mean = train_df['lagtime'].mean()
    train_df['lagtime'].fillna(lagtime_mean, inplace=True)

    train_df['lagtime'] = train_df['lagtime'] / (1000 * 3600)
    train_df.lagtime = train_df.lagtime.astype('float32')
    train_df['lagtime2'] = train_df.groupby(['user_id'])['timestamp'].shift(2)

    max_timestamp_u3 = train_df[['user_id', 'lagtime2']].groupby(['user_id'])\
        .agg(['max']).reset_index()
    max_timestamp_u3.columns = ['user_id', 'max_time_stamp3']
    max_timestamp_u3.user_id = max_timestamp_u3.user_id.astype('int32')

    train_df['lagtime2'] = train_df['timestamp'] - train_df['lagtime2']
    lagtime_mean2 = train_df['lagtime2'].mean()
    train_df['lagtime2'].fillna(lagtime_mean2, inplace=True)

    train_df['lagtime2'] = train_df['lagtime2'] / (1000 * 3600)
    train_df.lagtime2 = train_df.lagtime2.astype('float32')

    train_df['lagtime3'] = train_df.groupby('user_id')['timestamp'].shift(3)
    train_df['lagtime3'] = train_df['timestamp'] - train_df['lagtime3']
    lagtime_mean3 = train_df['lagtime3'].mean()
    train_df['lagtime3'].fillna(lagtime_mean3, inplace=True)
    train_df['lagtime3'] = train_df['lagtime3'] / (1000 * 3600)
    train_df.lagtime3 = train_df.lagtime3.astype('float32')

    train_df['timestamp'] = train_df['timestamp'] / (1000 * 3600)
    train_df.timestamp = train_df.timestamp.astype('float16')

    user_prior_question_elapsed_time =\
        train_df[['user_id', 'prior_question_elapsed_time']]\
        .groupby(['user_id']).tail(1)
    user_prior_question_elapsed_time.columns =\
        ['user_id', 'prior_question_elapsed_time']

    train_df['delta_prior_question_elapsed_time'] =\
        train_df.groupby('user_id')['prior_question_elapsed_time'].shift()
    train_df['delta_prior_question_elapsed_time'] =\
        train_df['prior_question_elapsed_time']\
        - train_df['delta_prior_question_elapsed_time']

    delta_prior_question_elapsed_time_mean =\
        train_df['delta_prior_question_elapsed_time'].mean()
    train_df['delta_prior_question_elapsed_time']\
        .fillna(delta_prior_question_elapsed_time_mean, inplace=True)
    train_df.delta_prior_question_elapsed_time =\
        train_df.delta_prior_question_elapsed_time.astype('int32')

    train_df['lag'] = train_df.groupby('user_id')[target].shift()

    cum = train_df.groupby('user_id')['lag'].agg(['cumsum', 'cumcount'])
    cum['cumsum'].fillna(0, inplace=True)

    train_df['user_correctness'] = cum['cumsum'] / cum['cumcount']
    train_df['user_correct_count'] = cum['cumsum']
    train_df['user_uncorrect_count'] = cum['cumcount'] - cum['cumsum']
    train_df.drop(columns=['lag'], inplace=True)

    train_df['user_correctness'].fillna(0.67, inplace=True)
    train_df.user_correctness = train_df.user_correctness.astype('float16')
    train_df.user_correct_count = train_df.user_correct_count.astype('int16')
    train_df.user_uncorrect_count =\
        train_df.user_uncorrect_count.astype('int16')

    del cum
    gc.collect()

    train_df.prior_question_had_explanation =\
        train_df.prior_question_had_explanation.astype('int8')

    explanation_agg = train_df.groupby('user_id')[
                'prior_question_had_explanation'].agg(['sum', 'count'])
    explanation_agg = explanation_agg.astype('int16')

    cum = train_df.groupby('user_id')[
        'prior_question_had_explanation'].agg(['cumsum', 'cumcount'])
    cum['cumcount'] = cum['cumcount'] + 1
    train_df['explanation_mean'] = cum['cumsum'] / cum['cumcount']
    train_df['explanation_true_count'] = cum['cumsum']
    train_df['explanation_false_count'] = cum['cumcount'] - cum['cumsum']

    train_df.explanation_mean = train_df.explanation_mean.astype('float16')
    train_df.explanation_true_count =\
        train_df.explanation_true_count.astype('int16')
    train_df.explanation_false_count =\
        train_df.explanation_false_count.astype('int16')

    del cum
    gc.collect()

    task_container_agg = train_df.groupby('task_container_id')[target]\
        .agg(['sum', 'count', 'var'])
    task_container_agg = task_container_agg.astype('float32')

    train_df['task_container_uncor_count'] = train_df['task_container_id']\
        .map(task_container_agg['count'] - task_container_agg['sum'])\
        .astype('int32')

    train_df['task_container_cor_count'] = train_df['task_container_id']\
        .map(task_container_agg['sum']).astype('int32')

    train_df['task_container_std'] = train_df['task_container_id']\
        .map(task_container_agg['var']).astype('float16')

    train_df['task_container_correctness'] = train_df['task_container_id']\
        .map(task_container_agg['sum'] / task_container_agg['count'])

    train_df.task_container_correctness =\
        train_df.task_container_correctness.astype('float16')

    return train_df


train_df = train_preprocessing(path)
