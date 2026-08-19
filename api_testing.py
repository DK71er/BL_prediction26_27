import requests
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
JSON_DIR = PROJECT_ROOT / "data" / "json" 

JSON_DIR.mkdir(parents=True, exist_ok=True)

season = 2026
base_url = "https://api.openligadb.de/getmatchdata/bl1"

for i in range(1, 35):
    response = requests.get(f"{base_url}/{season}/{i}")
    data = response.json()

    with open(JSON_DIR / f"matchday{i}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Saved matchday 1-34 as json in {JSON_DIR}!")