import requests
import json
from pathlib import Path
import pandas as pd

def get_json(base_url: str, season: int, JSON_DIR: str):
    for i in range(1, 35):
        response = requests.get(f"{base_url}/{season}/{i}")
        data = response.json()

        with open(JSON_DIR / f"matchday{i}.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved matchday 1-34 as json in {JSON_DIR}!")


def create_csv_from_json(JSON_DIR: str):
    df = pd.DataFrame(columns=('Team_Home', 'Team_Away', 'Date', 'Matchday_Home', 'Matchday_Away'))
    for i in range(1, 35):
        with open(JSON_DIR / f"matchday{i}.json", "r") as f:
            data = json.load(f)
        for j in range(len(data)):
            df.loc[j + len(data) * (i-1)] = [data[j]["team1"]["teamName"]] + [data[j]["team2"]["teamName"]] + [data[j]["matchDateTime"]] + [i] + [i]

    df['Date'] = pd.to_datetime(df["Date"]).dt.date
    df['Season'] = season
    df = df.set_index('Date').sort_values('Date')
    df.to_csv(JSON_DIR/ "new_matchday.csv")
    print(f"Saved csvs in {JSON_DIR}!")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent
    JSON_DIR = PROJECT_ROOT / "data" / "json" 
    JSON_DIR.mkdir(parents=True, exist_ok=True)


    season = 2026
    base_url = "https://api.openligadb.de/getmatchdata/bl1"

    create_csv_from_json(JSON_DIR)