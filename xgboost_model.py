from splitting_data import Xtrain, ytrain, Xval, yval, Xtest, ytest
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import root_mean_squared_error

def eval_model():
    print('Evaluating model ...')
    print()

    def featur_importances():
        imp_home = pd.Series(model_home.feature_importances_, index=Xtrain.columns).sort_values(ascending=False)
        imp_away = pd.Series(model_away.feature_importances_, index=Xtrain.columns).sort_values(ascending=False)
        print(f'Feature-importance home-model: {imp_home}')
        print(f'Feature-importance away-model: {imp_away}')

    def rmse_baseline_real():
        baseline_home = root_mean_squared_error(yval['GoalsFor_Home'], ([ytrain['GoalsFor_Home'].mean()] * len(yval)))
        rmse_home = root_mean_squared_error(yval['GoalsFor_Home'], model_home.predict(Xval))
        baseline_away = root_mean_squared_error(yval['GoalsFor_Away'], ([ytrain['GoalsFor_Away'].mean()] * len(yval)))
        rmse_away = root_mean_squared_error(yval['GoalsFor_Away'], model_away.predict(Xval))

        print(f"Baseline home-model rmse: {baseline_home:.3f}")
        print(f"Home-model rmse: {rmse_home:.3f}")
        print(f"Home-model outperforms baseline (mean) by {(baseline_home - rmse_home) / baseline_home * 100:.1f}%!")
        print()
        print(f"Baseline away-model rmse: {baseline_away:.3f}")
        print(f"Away-model rmse: {rmse_away:.3f}")
        print(f"Away-model outperforms baseline (mean) by {(baseline_away - rmse_away) / baseline_away * 100:.1f}%!")
        

    rmse_baseline_real()




params_model = {'objective': 'count:poisson',
                'n_estimators': 2000,
                'max_depth': 4,
                'learning_rate': 0.03,
                'early_stopping_rounds': 50,
                'enable_categorical': True}

print('Training model ...')
model_home = XGBRegressor(**params_model)
model_away = XGBRegressor(**params_model)

model_home.fit(Xtrain, ytrain['GoalsFor_Home'],
                eval_set=[(Xval, yval['GoalsFor_Home'])],
                verbose= 0
                )

model_away.fit(Xtrain, ytrain['GoalsFor_Away'],
                eval_set=[(Xval, yval['GoalsFor_Away'])],
                verbose=0
                )

eval_model()