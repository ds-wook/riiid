from optim.bayesian_optim import lgb_bayes

if __name__ == "__main__":

    bayesian_params = {
        'num_leaves': (300, 1000),
        'max_bin': (10, 400),
        'learning_rate': (0.01, 0.1),
    }
    lgb_bo = lgb_bayes(bayesian_params)
