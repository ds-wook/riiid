import optuna
from optuna.samplers import TPESampler
from optim.optuna_optim import objective

if __name__ == "__main__":
    sampler = TPESampler(seed=2020)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)
    print("  Params: ")
    params = study.best_params
    params["random_state"] = 2020
    print(params)
