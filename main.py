import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from create_future_features import future_features

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed"

matchday = 1
features = pd.read_csv(FEATURE_DIR / "features.csv")
future_features = pd.read_csv(FEATURE_DIR / f"future_features_md{matchday}.csv")

FEATURE_COLS_DROP = [
    'Date', 'Div',
    'Result_Home', 'GoalsForHT_Home', 'HTResult_Home', 'TeamShots_Home',
    'TeamShotsOnTarget_Home', 'TeamCorners_Home', 'TeamFouls_Home',
    'TeamYellow_Home', 'TeamRed_Home', 'PointsFor_Home',
    'Result_Away', 'GoalsForHT_Away', 'HTResult_Away', 'TeamShots_Away',
    'TeamShotsOnTarget_Away', 'TeamCorners_Away', 'TeamFouls_Away',
    'TeamYellow_Away', 'TeamRed_Away', 'PointsFor_Away',
]


def x_y_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[['GoalsFor_Home', 'GoalsFor_Away']]
    x = df.drop(columns=['GoalsFor_Home', 'GoalsFor_Away'] + FEATURE_COLS_DROP)
    return x, y


def align_categoricals(train_df: pd.DataFrame, future_df: pd.DataFrame, cat_cols: list[str]) -> None:
    for col in cat_cols:
        categories = pd.concat([train_df[col], future_df[col]]).astype('object').unique()
        cat_type = pd.CategoricalDtype(categories=categories)
        train_df[col] = train_df[col].astype(cat_type)
        future_df[col] = future_df[col].astype(cat_type)


Xtrain, ytrain = x_y_split(features)
Xfuture = future_features.drop(columns=[c for c in FEATURE_COLS_DROP if c in future_features.columns])
Xfuture = Xfuture[Xtrain.columns]  

cat_cols = Xtrain.select_dtypes(include=['object', 'string']).columns.tolist()
align_categoricals(Xtrain, Xfuture, cat_cols)

params_model_home = {'objective': 'count:poisson', 'n_estimators': 275, 'max_depth': 6,
                      'learning_rate': 0.01, 'subsample': 0.872, 'colsample_bytree': 0.605,
                      'min_child_weight': 1, 'gamma': 2.54, 'reg_alpha': 0.202,
                      'reg_lambda': 8.35, 'enable_categorical': True}

params_model_away = {'objective': 'count:poisson', 'n_estimators': 274, 'max_depth': 3,
                      'learning_rate': 0.034, 'subsample': 0.624, 'colsample_bytree': 0.622,
                      'min_child_weight': 2, 'gamma': 0.0133, 'reg_alpha': 0.449,
                      'reg_lambda': 2.53, 'enable_categorical': True}

print('Training model ...')
model_home = XGBRegressor(**params_model_home)
model_away = XGBRegressor(**params_model_away)

model_home.fit(Xtrain, ytrain['GoalsFor_Home'], verbose=0)
model_away.fit(Xtrain, ytrain['GoalsFor_Away'], verbose=0)

pred_home = model_home.predict(Xfuture)
pred_away = model_away.predict(Xfuture)

results = future_features[['Date', 'Team_Home', 'Team_Away']].copy()
results['Pred_GoalsFor_Home'] = pred_home.round(2)
results['Pred_GoalsFor_Away'] = pred_away.round(2)

if __name__ == "__main__":
    print(results.to_string(index=False))