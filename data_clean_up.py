from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

def make_file_paths(division: int) -> list[str]:
    files = []
    for i in range(16):
        start_y = 10 + i
        end_y = 11 + i
        files.append(f"{start_y}_{end_y}_D{division}.csv")

    return files

def concat_csvs(d_files):
    dfs = []
    for index, file in enumerate(d_files):
        df = pd.read_csv(DATA_DIR / file)
        df["Season"] = 2010 + int(index)
        dfs.append(df)

    d_all = pd.concat(dfs, ignore_index=True)

    return d_all


d1_files = make_file_paths(1)
d2_files = make_file_paths(2)

d1_all = concat_csvs(d1_files)
d2_all = concat_csvs(d2_files)

keep_cols = ['HomeTeam', 'AwayTeam', 'Div', 'Season', 'Date', 'FTR', 'FTHG', 'FTAG', 'HTHG', 'HTAG', 'HTR', 'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR' ]


d1_all_filtered = d1_all[keep_cols]
d2_all_filtered = d2_all[keep_cols]

d1_all_filtered = d1_all_filtered.dropna(subset=["HS"]) #filter the one game with NaNs

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

d1_all_filtered.to_csv(PROCESSED_DIR / "d1_clean.csv", index=False)
d2_all_filtered.to_csv(PROCESSED_DIR / "d2_clean.csv", index=False)