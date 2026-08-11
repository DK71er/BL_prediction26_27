from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def train_test_val_split(df: pd.DataFrame) -> pd.DataFrame:

    def x_y_split(df: pd.DataFrame) -> pd.DataFrame:
        y = df[['GoalsFor_Home', 'GoalsFor_Away']]
        x = df.drop(columns=['GoalsFor_Home', 'GoalsFor_Away'])

        return x, y

    train_mask = df['Season'] < 2023
    val_mask = df['Season'] == 2023
    test_mask = df['Season'] > 2023

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    Xtrain, ytrain = x_y_split(train_df)
    Xval, yval = x_y_split(val_df)
    Xtest, ytest = x_y_split(test_df)

    feature_cols_drop = ['Date', 'Result_Home', 'GoalsForHT_Home', 'HTResult_Home', 'TeamShots_Home', 'TeamShotsOnTarget_Home', 'TeamCorners_Home', 'TeamFouls_Home', 'TeamYellow_Home', 'TeamRed_Home', 'PointsFor_Home',
                         'Result_Away', 'GoalsForHT_Away', 'HTResult_Away', 'TeamShots_Away', 'TeamShotsOnTarget_Away', 'TeamCorners_Away', 'TeamFouls_Away', 'TeamYellow_Away', 'TeamRed_Away', 'PointsFor_Away']
    Xtrain = Xtrain.drop(columns=feature_cols_drop)
    Xval = Xval.drop(columns=feature_cols_drop)
    Xtest = Xtest.drop(columns=feature_cols_drop)

    return Xtrain, ytrain, Xval, yval, Xtest, ytest


features = pd.read_csv(DATA_DIR / "features.csv")

Xtrain, ytrain, Xval, yval, Xtest, ytest = train_test_val_split(features)