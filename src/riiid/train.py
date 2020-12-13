from optim.bayesian_optim import lgb_bayes
from sklearn.metrics import roc_auc_score

from model.pipeline_model import pipeline_lgb
from cleaning.dataset import train, target
from cleaning.dataset import X_train, X_valid, y_train, y_valid

if __name__ == "__main__":

    bayesian_params = {
        'learning_rate': (0.01, 0.5),
        'max_depth': (1, 30),
        'num_leaves': (10, 200),
        'colsample_bytree': (0.5, 1.0),
        'subsample': (0.1, 1.0)
    }
    lgb_bo = lgb_bayes(train, target, bayesian_params)

    params = {
        "learning_rate": max(min(lgb_bo['learning_rate'], 1), 0),
        'num_leaves': int(round(lgb_bo['num_leaves'])),
        'max_depth':  int(round(lgb_bo['max_depth'])),
        'colsample_bytree': max(min(lgb_bo['colsample_bytree'], 1), 0),
        'subsample': max(min(lgb_bo['subsample'], 1), 0)
    }

    lgb_model = pipeline_lgb(train, params)
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict_proba(X_valid)[:, 1]
    print(f'ROC_AUC: {roc_auc_score(y_valid, lgb_pred):.4f}')
