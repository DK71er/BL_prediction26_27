from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

def train_test_val_split(df: pd.DataFrame) -> pd.DataFrame:
    train_mask = df['Season'] < 2023
    val_mask = df['Season'] == 2023
    test_mask = df['Season'] > 2023

    train_df = df[train_mask]
    val_df = df[val_mask]
    test_df = df[test_mask]

    def x_y_split(df: pd.DataFrame) -> pd.DataFrame:
        y = df[['GoalsFor_Home', 'GoalsFor_Away']]
        x = df.drop(columns=['GoalsFor_Home', 'GoalsFor_Away'])

        return x, y

    Xtrain, ytrain = x_y_split(train_df)
    Xval, yval = x_y_split(val_df)
    Xtest, ytest = x_y_split(test_df)

    return Xtrain, ytrain, Xval, yval, Xtest, ytest


features = pd.read_csv(DATA_DIR / "features.csv")

Xtrain, ytrain, Xval, yval, Xtest, ytest = train_test_val_split(features)

#print(Xtrain.size + ytrain.size + Xval.size + yval.size + Xtest.size + ytest.size == features.size ) #test if no loss
