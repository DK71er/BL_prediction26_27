import pandas as pd
import numpy as np
from pathlib import Path
from feature_engineering import build_features

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

d1 = pd.read_csv(DATA_DIR / "d1_clean.csv")
d2 = pd.read_csv(DATA_DIR / "d2_clean.csv")

schedule = pd.read_csv(PROJECT_ROOT / "data" / "json" / "new_matchday.csv")
schedule['Date'] = pd.to_datetime(schedule['Date'])

# football-data.co.uk-Kurznamen (wie in d1_clean/d2_clean) <-> volle Vereinsnamen (wie in new_matchday.csv)
team_map = {
    'Bayern Munich': 'FC Bayern München', 'Ein Frankfurt': 'Eintracht Frankfurt', 'Leverkusen': 'Bayer 04 Leverkusen',
    'Freiburg': 'SC Freiburg', 'Union Berlin': '1. FC Union Berlin', 'Mainz': '1. FSV Mainz 05',
    "M'gladbach": 'Borussia Mönchengladbach', 'Hamburg': 'Hamburger SV', 'Werder Bremen': 'SV Werder Bremen',
    'Hoffenheim': 'TSG Hoffenheim', 'Augsburg': 'FC Augsburg', 'Stuttgart': 'VfB Stuttgart', 'Dortmund': 'Borussia Dortmund',
    'FC Koln': '1. FC Köln', 'Elversberg': 'SV 07 Elversberg', 'Schalke 04': 'FC Schalke 04',
    'Paderborn': 'SC Paderborn 07', 'RB Leipzig': 'RB Leipzig'
}
reverse_map = {full: short for short, full in team_map.items()}  # zurueck auf football-data-Kurznamen


def get_next_matchday(schedule: pd.DataFrame, d1: pd.DataFrame, d2: pd.DataFrame) -> pd.DataFrame:
    """Nur den naechsten noch nicht gespielten Spieltag aus dem kompletten Saisonplan
    rausfiltern. Wichtig: Rolling-Features fuer Spieltag N+1 brauchen die echten
    Ergebnisse von Spieltag N -> man kann nicht die ganze Saison auf einmal berechnen."""

    last_known_date = pd.concat([
        pd.to_datetime(d1['Date'], format='mixed', dayfirst=True),
        pd.to_datetime(d2['Date'], format='mixed', dayfirst=True)
    ]).max()

    upcoming = schedule[schedule['Date'] > last_known_date].copy()
    next_matchday_num = upcoming['Matchday_Home'].min()
    return upcoming[upcoming['Matchday_Home'] == next_matchday_num].copy()


def to_raw_schema(next_matchday: pd.DataFrame, raw_columns: pd.Index) -> pd.DataFrame:
    """Team_Home/Team_Away (volle Namen) -> HomeTeam/AwayTeam (Kurznamen), Div ergaenzen,
    fehlende Stat-Spalten (Ergebnis noch unbekannt) mit NaN auffuellen."""

    df = next_matchday.rename(columns={'Team_Home': 'HomeTeam', 'Team_Away': 'AwayTeam'})
    df['HomeTeam'] = df['HomeTeam'].replace(reverse_map)
    df['AwayTeam'] = df['AwayTeam'].replace(reverse_map)
    df['Div'] = 'D1'  # Annahme: new_matchday.csv enthaelt ausschliesslich 1.-Liga-Fixtures

    for col in raw_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[raw_columns]


def sanity_check_teams(next_matchday_raw: pd.DataFrame, d1: pd.DataFrame, d2: pd.DataFrame) -> None:
    """Warnt, falls ein Team aus dem Spielplan nach der Kurzname-Zuordnung nicht in der
    Historie auftaucht -> wuerde sonst still als 'neues' Team ohne Rolling-Historie laufen."""

    raw_teams = set(d1['HomeTeam']) | set(d1['AwayTeam']) | set(d2['HomeTeam']) | set(d2['AwayTeam'])
    new_teams = set(next_matchday_raw['HomeTeam']) | set(next_matchday_raw['AwayTeam'])
    unmatched = new_teams - raw_teams
    if unmatched:
        print(f"WARNUNG: unbekannte Teams nach Mapping, Rolling-Features werden NaN: {unmatched}")


next_matchday = get_next_matchday(schedule, d1, d2)
next_matchday_raw = to_raw_schema(next_matchday, d1.columns)
sanity_check_teams(next_matchday_raw, d1, d2)


# Future-Fixtures an BEIDE Divisionen anhaengen: fuer etablierte Teams entsteht in der
# "falschen" Division nur eine wertlose NaN-Zeile, die build_features() per Dedup wieder
# verwirft. Fuer Aufsteiger (z.B. Paderborn, Schalke, Elversberg) sorgt genau das dafuer,
# dass ihre Rolling-Features weiter aus der echten D2-Historie berechnet werden.
d1_future = pd.concat([d1, next_matchday_raw], ignore_index=True)
d2_future = pd.concat([d2, next_matchday_raw], ignore_index=True)
features_all = build_features(d1_future, d2_future)

future_features = features_all[features_all['Date'].isin(next_matchday['Date'])]

if __name__ == '__main__':
    rolling_cols = [c for c in future_features.columns if 'Rolling' in c]
    print(future_features[future_features['Team_Home'].isin(['Paderborn', 'Elversberg']) |
                       future_features['Team_Away'].isin(['Paderborn', 'Schalke 04', 'Elversberg'])]
      [['Team_Home', 'Team_Away'] + rolling_cols])
    print(future_features[['Date', 'Team_Home', 'Team_Away']])