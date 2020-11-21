# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
import pickle
import riiideducation


# %% [markdown]

path = '../../kaggle/input/riiid-test-answer-prediction/'
dtypes = {
        'timestamp': 'int64',
        'user_id': 'int32',
        'content_id': 'int16',
        'content_type_id': 'int8',
        'user_answer': 'int8',
        'answered_correctly': 'int8',
        'prior_question_elapsed_time': 'float32',
        'prior_question_had_explanation': 'boolean'
    }

train = pd.read_csv(
    path + 'train.csv',
    usecols=dtypes.keys(),
    dtype=dtypes
)

train = train.sort_values(['timestamp'], ascending=True)
questions = pd.read_csv(path + 'questions.csv')
lectures = pd.read_csv(path + 'lectures.csv')
print('Train shape:', train.shape)



# %%


print('Missing Value')
train.isnull().sum() / train.shape[0]


# %%

user = train['user_id'].value_counts().reset_index()
user.columns = ['user_id', 'count']
user['user_id'] = user['user_id'].astype(str) + '-'
user = user.sort_values(by='count', ascending=False).head(40)

plt.figure(figsize=(20, 12))
sns.barplot(x='count', y='user_id', data=user)
plt.title('Top 40 users by number of actions')
plt.show()


# %%

content = train['content_id'].value_counts().reset_index()
content.columns = ['content_id', 'count']
content['content_id'] = content['content_id'].astype(str) + '-'
content = content.sort_values(by='count', ascending=False).head(40)

plt.figure(figsize=(20, 12))
sns.barplot(x='count', y='content_id', data=content)
plt.title('Top 40 most useful content_ids')
plt.show()


# %%

content = train['content_id'].value_counts().reset_index()
content.columns = ['content_id', 'count']
content = content.sort_values('content_id')

plt.figure(figsize=(20, 12))
sns.lineplot(x='content_id', y='count', data=content)
plt.title('content_id action distribution')
plt.show()


# %%


content_type = train['content_type_id'].value_counts().reset_index()
content_type.columns = ['content_type_id', 'percent']
content_type['percent'] /= train.shape[0]

plt.figure(figsize=(12, 12))
plt.pie(
    content_type['percent'],
    labels=content_type['content_type_id'],
    autopct='%1.2f%%',
    startangle=90)
plt.title('Lecures & Questions')
plt.show()


# %%

task_container = train['task_container_id'].value_counts().reset_index()
task_container.columns = ['task_container_id', 'count']
task_container['task_container_id'] =\
    task_container['task_container_id'].astype(str) + '-'
task_container =\
    task_container.sort_values(by='count', ascending=False).head(40)

plt.figure(figsize=(20, 12))
sns.barplot(x='count', y='task_container_id', data=task_container)
plt.title('Top 40 most userful task_container_id')
plt.show()


# %%

task_container = train['task_container_id'].value_counts().reset_index()
task_container.columns = ['task_container_id', 'count']
task_container = task_container.sort_values('task_container_id')

plt.figure(figsize=(12, 8))
sns.lineplot(x='task_container_id', y='count', data=task_container)
plt.title('task_container_id action distribution')
plt.show()


# %%

user_answer = train['user_answer'].value_counts().reset_index()
user_answer.columns = ['user_answer', 'percent_of_answers']
user_answer['percent_of_answers'] /= train.shape[0]
user_answer = user_answer.sort_values(['percent_of_answers'])

plt.figure(figsize=(12, 8))
sns.barplot(x='user_answer', y='percent_of_answers', data=user_answer)
plt.show()


# %%

answered_correctly = train['answered_correctly'].value_counts().reset_index()
answered_correctly.columns = ['answered_correctly', 'percent_of_answers']
answered_correctly['percent_of_answers'] /= train.shape[0]

plt.figure(figsize=(12, 12))
plt.pie(
    answered_correctly['percent_of_answers'],
    labels=answered_correctly['answered_correctly'],
    autopct='%1.2f%%')
plt.title('Percent of correct answer')
plt.show()


# %%

plt.figure(figsize=(20, 12))
sns.distplot(x=train['prior_question_elapsed_time'], kde=False)
plt.title('prior_question_elapse_time distribution')
plt.show()


# %%


train = train.loc[train['answered_correctly'] != -1].reset_index(drop=True)
train = train.drop(['timestamp', 'content_type_id'], axis=1)
train['prior_question_had_explanation'] =\
    train['prior_question_had_explanation'].fillna(value=False).astype(bool)
train.head()


# %%


agg_dict = {'answered_correctly': ['mean', 'count']}
user_answers_df =\
     train.groupby('user_id').agg(agg_dict).copy()
user_answers_df.columns = ['mean_user_accuracy', 'questions_answered']
user_answers_df


# %%


content_answers_df =\
    train.groupby('content_id').agg(agg_dict).copy()
content_answers_df.columns = ['mean_accuracy', 'question_asked']
content_answers_df


# %%


train = train.iloc[90000000:, :]


# %%

train = train.merge(user_answers_df, on='user_id', how='left')
train = train.merge(content_answers_df, how='left', on='content_id')
train.head()


# %%
train.fillna(0.5, inplace=True)
# %%


le = LabelEncoder()
train['prior_question_had_explanation'] =\
    le.fit_transform(train['prior_question_had_explanation'])
train = train.sort_values(['user_id'])


# %%


target = train['answered_correctly']
columns = ['mean_user_accuracy', 'questions_answered',
           'mean_accuracy', 'question_asked',
           'prior_question_had_explanation']
train_x = train[columns]


# %%
del train
# %%

scores = []
feature_importance = pd.DataFrame()
models = []


# %%


params = {
    'num_leaves': 32,
    'max_bin': 300,
    'objective': 'binary',
    'max_depth': 13,
    'learning_rate': 0.02,
    'boosting_type': 'gbdt',
    'metric': 'auc'
    }

columns = ['mean_user_accuracy', 'questions_answered',
           'mean_accuracy', 'question_asked']
folds = StratifiedKFold(n_splits=5, shuffle=False)

for fold_n, (train_idx, valid_idx) in enumerate(folds.split(train_x, target)):
    X_train, X_valid = train_x[columns].iloc[train_idx], train_x[columns].iloc[valid_idx]
    y_train, y_valid = target.iloc[train_idx], target.iloc[valid_idx]
    model = LGBMClassifier(**params, n_estimators=1000, n_jobs=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_valid, y_valid)],
              eval_metric='auc',
              verbose=1000,
              early_stopping_rounds=10)
    score = max(model.evals_result_['valid_1']['auc'])
    models.append(model)
    scores.append(score)

    fold_importance = pd.DataFrame()
    fold_importance['feature'] = columns
    fold_importance['importance'] = model.feature_importances_
    fold_importance['fold'] = fold_n + 1
    feature_importance =\
        pd.concat([feature_importance, fold_importance], axis=0)
 

# %%

print(f'CV mean score: {np.mean(scores):.4f}, std: {np.std(scores):.4f}')


# %%
feature_importance['importance'] /= 1
cols = feature_importance[['feature', 'importance']].groupby('feature').mean().sort_values(by='importance', ascending=False)[:50].index
best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(figsize=(16, 12))
sns.barplot(x='importance', y='feature', data=best_features.sort_values(by='importance', ascending=False))
plt.show()
# %%

del train_x
# %%
del target

user_answers_df.to_csv('../../res/user_answers_df.csv')
content_answers_df.to_csv('../../res/content_answers_df.csv')
