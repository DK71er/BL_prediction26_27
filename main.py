import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FEATURE_DIR = PROJECT_ROOT / "data" / "processed"
NEW_MATCHDAY_DIR = PROJECT_ROOT / "data" / "json"

features_final = pd.read_csv(FEATURE_DIR / "features.csv")
