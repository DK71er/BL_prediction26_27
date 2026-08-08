import pandas as pd
from pathlib import Path

def rename_cols(df: pd.DataFrame):
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

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

d1 = pd.read_csv(DATA_DIR / "d1_clean.csv")
d2 = pd.read_csv(DATA_DIR / "d2_clean.csv")

d1_big = rename_cols(d1)
d2_big = rename_cols(d2)
print(d1_big.shape[0] == 2 * d1.shape[0])
print(d1_big['Team'].nunique())
print(d1_big[['Team','Opponent','Venue']].head())