import numpy as np
import pandas as pd
from splitting_data import Xtrain, ytrain, Xval, yval, Xtest, ytest
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.model_selection import TimeSeriesSplit


def find_params_model(venue: str, search_params: dict):
    cv = TimeSeriesSplit(n_splits=3)
    model = XGBRegressor(objective = 'count:poisson')

    search = RandomizedSearchCV(
        model, param_distributions=search_params,
        scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1,
        verbose=1, random_state=42, n_iter=50
    )

    search.fit(Xtrain, ytrain[f'GoalsFor_{venue}'])

    print(f'Best result for {venue} model:')
    print(search.best_params_)
    print(search.best_estimator_)
    print(search.best_score_)


param_dist = {
    'max_depth': randint(5, 9),
    'learning_rate': uniform(0.05, 0.15),      # 0.05–0.20
    'n_estimators': randint(250, 450),
    'subsample': uniform(0.6, 0.4),            # 0.6–1.0
    'colsample_bytree': uniform(0.6, 0.4),     # 0.6–1.0
    'min_child_weight': randint(1, 11),
    'gamma': uniform(0, 5),
    'reg_alpha': uniform(0, 2),
    'reg_lambda': uniform(0, 10),
    }

venue_options = ['Home', 'Away']#choose v[0] for home & v[1] for away

find_params_model(venue_options[0], param_dist)
find_params_model(venue_options[1], param_dist)