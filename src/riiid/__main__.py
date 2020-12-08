from optim.bayesian_optim import lgb_bayes
from cleaning.preprocessing import data_preprocessing


if __name__ == "__main__":

    path = '../../kaggle/input/riiid-test-answer-prediction/'
    train_x, target_y = data_preprocessing(path)
    print(f'train_x shape:{train_x.shape}')
    bayesian_params = {
        'num_leaves': (50, 500),
        'max_bin': (300, 850)
    }

    lgb_bo = lgb_bayes(bayesian_params)
