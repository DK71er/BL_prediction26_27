from splitting_data import Xtrain, ytrain, Xval, yval, Xtest, ytest
from xgboost import XGBRegressor
import pandas as pd
from sklearn.metrics import root_mean_squared_error

def eval_model():
    print('Evaluating model ...')


params_model = {'objective': 'count:poisson',
                'n_estimators': 2000,
                'max_depth': 4,
                'learning_rate': 0.03,
                'early_stopping_rounds': 50}

print('Training model ...')
model_home = XGBRegressor(**params_model)
model_away = XGBRegressor(**params_model)

model_home.fit(Xtrain, ytrain['GoalsFor_Home'],
                eval_set=[(Xval, yval['GoalsFor_Home'])],
                )

model_away.fit(Xtrain, ytrain['GoalsFor_Away'],
                eval_set=[(Xval, yval['GoalsFor_Away'])])

imp_home = pd.Series(model_home.feature_importances_, index=Xtrain.columns).sort_values(ascending=False)
imp_away = pd.Series(model_away.feature_importances_, index=Xtrain.columns).sort_values(ascending=False)

#print(imp_home)
#print(imp_away)

#print(model_home.predict(Xval).mean())
baseline_pred = [ytrain['GoalsFor_Home'].mean()] * len(ytest)
rmse_baseline = root_mean_squared_error(ytest['GoalsFor_Home'], baseline_pred)

print(baseline_pred)
print(rmse_baseline)