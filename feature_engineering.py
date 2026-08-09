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
    

    df = add_points(df)

    return df

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

d1 = pd.read_csv(DATA_DIR / "d1_clean.csv")
d2 = pd.read_csv(DATA_DIR / "d2_clean.csv")

d1_big = rename_cols(d1)
d2_big = rename_cols(d2)

d1_big = add_feature(d1_big)
d2_big = add_feature(d2_big)

d1_big = d1_big.set_index(['Team', 'Venue']).sort_index()
d2_big = d2_big.set_index(['Team', 'Venue']).sort_index()