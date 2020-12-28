from optim.bayesian_optim import lgb_bayes

if __name__ == "__main__":

    bayesian_params = {
        "num_leaves": (10, 800),
        "max_bin": (100, 400),
        "max_depth": (50, 800),
    }
    lgb_bo = lgb_bayes(bayesian_params)
