# %%


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import riiideducation


# %%


# Read in data
dtypes = {
    "row_id": "int64",
    "timestamp": "int64",
    "user_id": "int32",
    "content_id": "int16",
    "content_type_id": "int8",
    "task_container_id": "int16",
    "user_answer": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
    "prior_question_had_explanation": "boolean"
}

path = '../../riiid-test-answer-prediction/'

train = pd.read_csv(
    path + 'train.csv',
    low_memory=False,

    nrows=10**6,
    dtype=dtypes)
train.head()


# %%


train.info()


# %%


train.isna().sum()

