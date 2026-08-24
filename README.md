# BL Tipper
ML model for predicting the 2026/27 Bundesliga season

# Prediction of current matches:

<!-- TIPPS_START -->
### Raw predictions

| Date | Home | Home Goals (Pred) | Away Goals (Pred) | Away |
|---|---|---|---|---|
| 2026-08-28 | **Bayern Munich** | 2.75 | 1.12 | **Stuttgart** |
| 2026-08-29 | **Union Berlin** | 1.71 | 1.19 | **Ein Frankfurt** |
| 2026-08-29 | **Mainz** | 1.73 | 1.05 | **Paderborn** |
| 2026-08-29 | **Dortmund** | 2.35 | 1.01 | **Hamburg** |
| 2026-08-29 | **FC Koln** | 1.69 | 1.66 | **Hoffenheim** |
| 2026-08-29 | **RB Leipzig** | 2.16 | 0.94 | **M'gladbach** |
| 2026-08-29 | **Elversberg** | 1.59 | 1.91 | **Leverkusen** |
| 2026-08-30 | **Freiburg** | 1.74 | 1.36 | **Werder Bremen** |
| 2026-08-30 | **Augsburg** | 1.56 | 1.45 | **Schalke 04** |

### Tipp

Rounding rule: if the predicted goal difference is below 0.2, the match is called a draw (both scores rounded to the average). Otherwise each predicted score is rounded independently. This is a simple heuristic, not a calibrated model — a probability-based approach (Poisson score matrix) is a possible future improvement.

| Date | Home | Tipp | Away |
|---|---|---|---|
| 2026-08-28 | **Bayern Munich** | 3:1 | **Stuttgart** |
| 2026-08-29 | **Union Berlin** | 2:1 | **Ein Frankfurt** |
| 2026-08-29 | **Mainz** | 2:1 | **Paderborn** |
| 2026-08-29 | **Dortmund** | 2:1 | **Hamburg** |
| 2026-08-29 | **FC Koln** | 2:2 | **Hoffenheim** |
| 2026-08-29 | **RB Leipzig** | 2:1 | **M'gladbach** |
| 2026-08-29 | **Elversberg** | 2:2 | **Leverkusen** |
| 2026-08-30 | **Freiburg** | 2:1 | **Werder Bremen** |
| 2026-08-30 | **Augsburg** | 2:2 | **Schalke 04** |
<!-- TIPPS_END -->

## Status
Finished data preparation. Working on building the model.


### To-Do: Feature Engineering

- [X] Build long format (one row per team, home and away perspective separated)
- [X] Sort chronologically by `Date` per team
- [X] Compute rolling features (using `shift(1)` + `rolling(window=n)`)
- [X] Form (points from last 5 games)
- [X] Avg. goals scored / conceded
- [X] Avg. shots / corners / fouls / cards (D1 throughout, D2 from 2017 onward)
- [X] Table position feature (cumulative points before each matchday, per season)
- [X] Merge back into wide format (home rolling stats + away rolling stats per match)
- [X] Merge D1 + D2 (with `Div` as context feature)
- [X] Final NaN check (drop/handle first games per season with no rolling history)
- [X] Save as `data/processed/features.csv`

### To-Do: Modeling

- [X] Time-based train/test split instead of random (e.g. seasons through 2023 = train, 2024 = val, 2025/26 = test)
- [X] Define `FEATURE_COLS` allowlist (pre-match features only, no raw match stats as X) — needed for the ML step
- [X] ML model (XGBoost)
- [X] Optimize hyperparameters using Optuna

### Automatization

- [ ] Read newest matchday in + put last matchday to train
- [ ] Evaluate last matchday and post results on Github

### Extra:
- [ ] Implement Dixon-Coles-Model
- [ ] Evaluation: does ML actually beat Dixon-Coles? 