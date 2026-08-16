import optuna
from optuna.pruners import MedianPruner
from splitting_data import Xtrain, ytrain, Xval, yval, Xtest, ytest
from xgboost import XGBRegressor


def make_objective(venue: str):
    target_train = ytrain[f'GoalsFor_{venue}']
    target_val = yval[f'GoalsFor_{venue}']

    def objective(trial):
        params = {
            'objective': 'count:poisson',
            'tree_method': 'hist',          # deutlich schneller auf CPU
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
            'n_estimators': 500,            # hoch angesetzt, Early Stopping regelt die echte Anzahl
        }

        model = XGBRegressor(**params, early_stopping_rounds=20)
        model.fit(
            Xtrain, target_train,
            eval_set=[(Xval, target_val)],
            verbose=False,
        )

        trial.set_user_attr('best_iteration', model.best_iteration)
        return model.best_score  # RMSE des besten Iterationsschritts

    return objective


def find_params_model(venue: str, n_trials: int = 40):
    study = optuna.create_study(
        direction='minimize',
        pruner=MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(make_objective(venue), n_trials=n_trials, show_progress_bar=True)

    print(f'\nBest result for {venue} model:')
    print('Params:', study.best_params)
    print('Best n_estimators (early stopping):', study.best_trial.user_attrs['best_iteration'])
    print('Best RMSE:', study.best_value)

    return study


venue_options = ['Home', 'Away']  # v[0] = Home, v[1] = Away

if __name__ == '__main__':
    study_home = find_params_model(venue_options[0])
    study_away = find_params_model(venue_options[1])