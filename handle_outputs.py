import pandas as pd
from pathlib import Path
from main import results

print(results)

def format_readme_table(results: pd.DataFrame) -> str:
    lines = ["| Datum | Heim | Heimtore Pred | Auswärtstore Pred | Auswärts |", "|---|---|---|---|---|"]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['Date']} | **{row['Team_Home']}** | "
            f"{row['Pred_GoalsFor_Home']:.2f} | {row['Pred_GoalsFor_Away']:.2f} | "
            f"**{row['Team_Away']}** |"
        )
    return "\n".join(lines)


def update_readme(readme_path: Path, table_md: str) -> None:
    content = readme_path.read_text(encoding='utf-8')
    start_marker, end_marker = "<!-- TIPPS_START -->", "<!-- TIPPS_END -->"
    block = f"{start_marker}\n{table_md}\n{end_marker}"

    if start_marker in content:
        pre, rest = content.split(start_marker, 1)
        _, post = rest.split(end_marker, 1)
        content = pre + block + post
    else:
        content += f"\n\n{block}\n"

    readme_path.write_text(content, encoding='utf-8')

table_md = format_readme_table(results)
update_readme(Path("README.md"), table_md)