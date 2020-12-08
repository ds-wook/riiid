import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Dict, Union
np.seterr(divide='ignore', invalid='ignore')


def pipeline_lgb(
        train: pd.DataFrame,
        params: Dict[str, Union[float, int]]) -> LGBMClassifier:
    numeric_features = ['prior_question_elapsed_time', 'count',
                        'user_percent_correct', 'question_percent_correct']
    numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer()),
            ('scaler', StandardScaler())])

    categorical_features = ['prior_question_had_explanation']
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    lgb_model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LGBMClassifier(**params))])

    return lgb_model
