# BL Tipper
ML model for predicting the 2026/27 Bundesliga season

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

### To-Do: Modeling

- [X] Time-based train/test split instead of random (e.g. seasons through 2023 = train, 2024 = val, 2025/26 = test)
- [ ] Baseline: Dixon-Coles model (Poisson regression on home/away goal expectancy, with time decay)
- [X] Define `FEATURE_COLS` allowlist (pre-match features only, no raw match stats as X) — needed for the ML step
- [ ] ML model (XGBoost/Gradient Boosting on Elo, form, xG) — compare against Dixon-Coles baseline
- [ ] Decide final NaN handling (drop vs. model-native)
- [ ] Evaluation: does ML actually beat Dixon-Coles? (this comparison is the portfolio story)