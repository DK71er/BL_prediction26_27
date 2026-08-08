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
        df["season"] = 2010 + int(index)
        dfs.append(df)

    d_all = pd.concat(dfs, ignore_index=True)

    return d_all


d1_files = make_file_paths(1)
d2_files = make_file_paths(2)

d1_all = concat_csvs(d1_files)
d2_all = concat_csvs(d2_files)

print(d1_all.head())