import subprocess
import pandas as pd
from pathlib import Path
from main import results


def make_tipp(pred_home: float, pred_away: float, draw_threshold: float = 0.2) -> str:
    if abs(pred_home - pred_away) < draw_threshold:
        avg = round((pred_home + pred_away) / 2)
        return f"{avg}:{avg}"
    return f"{round(pred_home)}:{round(pred_away)}"


results['Tipp'] = results.apply(
    lambda row: make_tipp(row['Pred_GoalsFor_Home'], row['Pred_GoalsFor_Away']), axis=1
)


def format_raw_table(results: pd.DataFrame) -> str:
    lines = ["| Date | Home | Home Goals (Pred) | Away Goals (Pred) | Away |", "|---|---|---|---|---|"]
    for _, row in results.iterrows():
        lines.append(
            f"| {row['Date']} | **{row['Team_Home']}** | "
            f"{row['Pred_GoalsFor_Home']:.2f} | {row['Pred_GoalsFor_Away']:.2f} | "
            f"**{row['Team_Away']}** |"
        )
    return "\n".join(lines)


def format_tipp_table(results: pd.DataFrame) -> str:
    lines = ["| Date | Home | Tipp | Away |", "|---|---|---|---|"]
    for _, row in results.iterrows():
        lines.append(f"| {row['Date']} | **{row['Team_Home']}** | {row['Tipp']} | **{row['Team_Away']}** |")
    return "\n".join(lines)


TIPP_RULE_EXPLANATION = (
    "Rounding rule: if the predicted goal difference is below 0.2, the match is "
    "called a draw (both scores rounded to the average). Otherwise each predicted "
    "score is rounded independently. This is a simple heuristic, not a calibrated "
    "model — a probability-based approach (Poisson score matrix) is a possible future improvement."
)


def build_readme_block(results: pd.DataFrame) -> str:
    return (
        "### Raw predictions\n\n"
        f"{format_raw_table(results)}\n\n"
        "### Tipp\n\n"
        f"{TIPP_RULE_EXPLANATION}\n\n"
        f"{format_tipp_table(results)}"
    )


def update_readme(readme_path: Path, block_content: str) -> None:
    content = readme_path.read_text(encoding='utf-8')
    start_marker, end_marker = "<!-- TIPPS_START -->", "<!-- TIPPS_END -->"
    block = f"{start_marker}\n{block_content}\n{end_marker}"

    if start_marker in content:
        pre, rest = content.split(start_marker, 1)
        _, post = rest.split(end_marker, 1)
        content = pre + block + post
    else:
        content += f"\n\n{block}\n"

    readme_path.write_text(content, encoding='utf-8')


readme_path = Path("README.md")
update_readme(readme_path, build_readme_block(results))

subprocess.run(["git", "add", "README.md"], check=True)
diff = subprocess.run(["git", "diff", "--cached", "--quiet"])
if diff.returncode == 0:
    print("Keine Änderungen an README.md, kein Commit.")
else:
    subprocess.run(["git", "commit", "-m", "new matchday preds"], check=True)
    subprocess.run(["git", "push"], check=True)