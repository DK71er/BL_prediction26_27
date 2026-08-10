import pandas as pd
import numpy as np
from pathlib import Path

def rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    mapping_home = {'HomeTeam': 'Team', 'AwayTeam': 'Opponent', 'FTR': 'Result', 'FTHG': 'GoalsFor', 'FTAG': 'GoalsAgainst', 'HTHG': 'GoalsForHT', 'HTAG': 'GoalsAgainstHT', 'HTR': 'HTResult', 'HS': 'TeamShots', 'AS': 'OpponentShots', 'HST': 'TeamShotsOnTarget', 'AST': 'OpponentShotsOnTarget', 'HC': 'TeamCorners', 'AC': 'OpponentCorners', 'HF': 'TeamFouls', 'AF': 'OpponentFouls', 'HY': 'TeamYellow', 'AY': 'OpponentYellow', 'HR': 'TeamRed', 'AR': 'OpponentRed'}
    mapping_away = {'HomeTeam': 'Opponent', 'AwayTeam': 'Team', 'FTR': 'Result', 'FTHG': 'GoalsAgainst', 'FTAG': 'GoalsFor', 'HTHG': 'GoalsAgainstHT', 'HTAG': 'GoalsForHT', 'HTR': 'HTResult', 'HS': 'OpponentShots', 'AS': 'TeamShots', 'HST': 'OpponentShotsOnTarget', 'AST': 'TeamShotsOnTarget', 'HC': 'OpponentCorners', 'AC': 'TeamCorners', 'HF': 'OpponentFouls', 'AF': 'TeamFouls', 'HY': 'OpponentYellow', 'AY': 'TeamYellow', 'HR': 'OpponentRed', 'AR': 'TeamRed'}
    df_home = df.rename(columns=mapping_home)
    df_away = df.rename(columns=mapping_away)

    df_home['Venue'] = 'Home'
    df_away['Venue'] = 'Away'

    result_map = {'H':'A','A':'H','D':'D'}
    df_away['Result'] = df_away['Result'].map(result_map)
    df_away['HTResult'] = df_away['HTResult'].map(result_map)

    df = pd.concat([df_home, df_away], ignore_index=True)
    return df

def add_feature(df: pd.DataFrame) -> pd.DataFrame:

    def add_points(df: pd.DataFrame) -> pd.DataFrame:
        conditions = [
            (df['GoalsFor'] > df['GoalsAgainst']),
            (df['GoalsFor'] == df['GoalsAgainst']),
            (df['GoalsFor'] < df['GoalsAgainst'])
        ]

        choices = [3, 1, 0]

        df['PointsFor'] = np.select(conditions, choices)

        return df

    def add_rolling(df: pd.DataFrame, feature: str, rolling_size: int, agg_func: str) -> pd.DataFrame:

        new_col = f"{feature}Rolling{rolling_size}"

        df[new_col] = df.groupby('Team')[feature].transform(lambda x: x.shift(1).rolling(rolling_size).agg(agg_func))

        return df

    def add_table(df: pd.DataFrame):
        df['Matchday'] = df.groupby(['Team', 'Season']).cumcount() + 1
        df['CumPoints'] = df.groupby(['Team', 'Season'])['PointsFor'].transform(lambda x: x.shift(1).cumsum())
        df['TableRank'] = df.groupby(['Season', 'Matchday'])['CumPoints'].rank(method='min', ascending=False)

        return df

    df['GoalPerShot'] = df['GoalsFor'] / df['TeamShots']
    df['GoalsFor/GoalsAgainst'] = df['GoalsFor'] / df['GoalsAgainst']
    df = df.replace([np.inf, -np.inf], np.nan)

    df = add_points(df)
    df = add_rolling(df, 'PointsFor', 5, 'sum')
    df = add_rolling(df, 'TeamCorners', 5, 'mean')
    df = add_rolling(df, 'TeamRed', 5, 'sum')
    df = add_rolling(df, 'TeamYellow', 5, 'sum')
    df = add_rolling(df, 'TeamShotsOnTarget', 5, 'mean')
    df = add_rolling(df, 'TeamShots', 5, 'mean')
    df = add_rolling(df, 'GoalsFor', 5, 'mean')
    df = add_rolling(df, 'GoalsAgainst', 5, 'mean')
    df = add_rolling(df, 'TeamFouls', 5, 'mean')

    df = add_rolling(df, 'GoalPerShot', 5, 'mean')
    df = df.drop(columns=['GoalPerShot'])
    df = add_rolling(df, 'GoalsFor/GoalsAgainst', 5, 'mean')
    df = df.drop(columns=['GoalsFor/GoalsAgainst'])

    df['PointsLastMatch'] = df.groupby('Team')['PointsFor'].transform(lambda x: x.shift(1))
    

    df = add_table(df)

    return df

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

d1 = pd.read_csv(DATA_DIR / "d1_clean.csv")
d2 = pd.read_csv(DATA_DIR / "d2_clean.csv")

d1_big = rename_cols(d1)
d2_big = rename_cols(d2)

d1_big['Date'] = pd.to_datetime(d1_big['Date'], format='mixed', dayfirst=True)# convert Date to datetime
d2_big['Date'] = pd.to_datetime(d2_big['Date'], format='mixed', dayfirst=True)

d1_big = d1_big.set_index(['Team', 'Date']).sort_values(['Team', 'Date'])#add multiindexing
d2_big = d2_big.set_index(['Team', 'Date']).sort_values(['Team', 'Date'])

d1_big = add_feature(d1_big)
d2_big = add_feature(d2_big)
