import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

features = pd.read_csv(PROJECT_ROOT / "data" / "processed" / 'features.csv')
new_matchday = pd.read_csv(PROJECT_ROOT / "data" / "json" / 'new_matchday.csv')


def map_teams(df: pd.DataFrame) -> pd.DataFrame:
    team_map = { 
        'Bayern Munich': 'FC Bayern München', 'Ein Frankfurt': 'Eintracht Frankfurt', 'Leverkusen': 'Bayer 04 Leverkusen',
        'Freiburg': 'SC Freiburg', 'Union Berlin': '1. FC Union Berlin', 'Mainz': '1. FSV Mainz 05',
        "M'gladbach": 'Borussia Mönchengladbach', 'Hamburg': 'Hamburger SV', 'Werder Bremen': 'SV Werder Bremen',
        'Hoffenheim': 'TSG Hoffenheim', 'Augsburg': 'FC Augsburg', 'Stuttgart': 'VfB Stuttgart', 'Dortmund': 'Borussia Dortmund', 'FC Koln': '1. FC Köln',
        'Elversberg': 'SV 07 Elversberg', 'Schalke 04': 'FC Schalke 04', 'Paderborn': 'SC Paderborn 07', 'RB Leipzig': 'RB Leipzig'
    }

    df['Team_Home'] = df['Team_Home'].replace(team_map)
    df['Team_Away'] = df['Team_Away'].replace(team_map)
    
    return df

features = map_teams(features)

