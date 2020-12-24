import gc

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from data.dataset import train_df

np.seterr(divide='ignore', invalid='ignore')
path = '../../input/riiid-test-answer-prediction/'


def gettags(
        tags: str,
        num: int) -> str:
    tags_splits = tags.split(" ")
    result = ' '.join([
        t for t in tags_splits if 32 * num <= int(t) < 32 * (num + 1)])

    return result


def questions_feature_engine(path: str) -> pd.DataFrame:
    target = 'answered_correctly'
    questions_df = pd.read_csv(
        path + 'questions.csv',
        usecols=[0, 1, 3, 4],
        dtype={
            'question_id': 'int16',
            'bundle_id': 'int16',
            'part': 'int8',
            'tags': 'str'})

    bundle_agg = questions_df\
        .groupby(['bundle_id'])['question_id'].agg(['count'])

    questions_df['content_sub_bundle'] =\
        questions_df['bundle_id'].map(bundle_agg['count']).astype('int8')
    questions_df['tags'].fillna('188', inplace=True)

    for num in range(6):
        questions_df['tags' + str(num)] =\
            questions_df['tags'].apply(lambda row: gettags(row, num))

        le = LabelEncoder()
        le.fit(np.unique(questions_df['tags' + str(num)].values))

        questions_df['tags' + str(num)] =\
            questions_df[['tags' + str(num)]].apply(le.transform)

    questions_df_dict = {
        'tags0': 'int8',
        'tags1': 'int8',
        'tags2': 'int8',
        'tags3': 'int8',
        'tags4': 'int8',
        'tags5': 'int8',
    }

    questions_df = questions_df.astype(questions_df_dict)
    questions_df.drop(columns=['tags'], inplace=True)

    questions_df['part_bundle_id'] =\
        questions_df['part'] * 100000 + questions_df['bundle_id']
    questions_df.part_bundle_id = questions_df.part_bundle_id.astype('int32')

    questions_cmnts = pd.read_csv(
        path + 'question_cmnts.csv',
        usecols=[1, 2],
        dtype={'question_id': 'int16', 'community': 'int8'})

    questions_df = pd.merge(questions_df, questions_cmnts,
                            on='question_id', how='left', right_index=True)

    questions_df.rename(columns={'question_id': 'content_id'}, inplace=True)

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

    questions_df = pd.merge(questions_df, content_explation_agg,
                            on='content_id', how='left', right_index=True)

    del content_explation_agg

    content_agg = train_df.groupby('content_id')[target]\
        .agg(['sum', 'count', 'var'])

    questions_df['content_correctness'] = questions_df['content_id']\
        .map(content_agg['sum'] / content_agg['count'])
    questions_df.content_correctness =\
        questions_df.content_correctness.astype('float16')
    questions_df['content_correctness_std'] =\
        questions_df['content_id'].map(content_agg['var'])
    questions_df.content_correctness_std =\
        questions_df.content_correctness_std.astype('float16')
    questions_df['content_uncorrect_count'] =\
        questions_df['content_id'].map(
            content_agg['count'] - content_agg['sum']).astype('int32')
    questions_df['content_correct_count'] =\
        questions_df['content_id'].map(content_agg['sum']).astype('int32')

    content_elapsed_time_agg =\
        train_df.groupby('content_id')['prior_question_elapsed_time']\
        .agg(['mean'])
    content_had_explanation_agg =\
        train_df.groupby('content_id')['prior_question_had_explanation']\
        .agg(['mean'])

    questions_df['content_elapsed_time_mean'] =\
        questions_df['content_id'].map(content_elapsed_time_agg['mean'])
    questions_df.content_elapsed_time_mean =\
        questions_df.content_elapsed_time_mean.astype('float16')
    questions_df['content_had_explanation_mean'] =\
        questions_df['content_id'].map(content_had_explanation_agg['mean'])
    questions_df.content_had_explanation_mean =\
        questions_df.content_had_explanation_mean.astype('float16')
    del content_elapsed_time_agg
    del content_had_explanation_agg
    gc.collect()

    part_agg = questions_df\
        .groupby('part')['content_correctness'].agg(['mean', 'var'])
    questions_df['part_correctness_mean'] =\
        questions_df['part'].map(part_agg['mean'])
    questions_df['part_correctness_std'] =\
        questions_df['part'].map(part_agg['var'])
    questions_df.part_correctness_mean =\
        questions_df.part_correctness_mean.astype('float16')
    questions_df.part_correctness_std =\
        questions_df.part_correctness_std.astype('float16')

    part_agg = questions_df\
        .groupby('part')['content_uncorrect_count'].agg(['sum'])
    questions_df['part_uncor_count'] = questions_df['part']\
        .map(part_agg['sum']).astype('int32')

    part_agg = questions_df\
        .groupby('part')['content_correct_count'].agg(['sum'])
    questions_df['part_cor_count'] =\
        questions_df['part'].map(part_agg['sum']).astype('int32')

    bundle_agg =\
        questions_df.groupby('bundle_id')['content_correctness'].agg(['mean'])
    questions_df['bundle_correctness_mean'] =\
        questions_df['bundle_id'].map(bundle_agg['mean'])
    questions_df.bundle_correctness_mean =\
        questions_df.bundle_correctness_mean.astype('float16')

    del content_agg
    del bundle_agg
    del part_agg
    gc.collect()

    return questions_df


questions_df = questions_feature_engine(path)
