# BL Tipper
ML model for predicting the 2026/27 Bundesliga season

## Status
Working on data preparation.

### To-Do: Feature Engineering

- [X] Build long format (one row per team, home and away perspective separated)
- [X] Sort chronologically by `Date` per team
- [X] Compute rolling features (using `shift(1)` + `rolling(window=n)`)
- [X] Form (points from last 5 games)
- [X] Avg. goals scored / conceded
- [X] Avg. shots / corners / fouls / cards (D1 throughout, D2 from 2017 onward)
- [X] Table position feature (cumulative points before each matchday, per season)
- [ ] Merge back into wide format (home rolling stats + away rolling stats per match)
- [ ] Merge D1 + D2 (with `Div` as context feature)
- [ ] Final NaN check (drop/handle first games per season with no rolling history)
- [ ] Save as `data/processed/features.csv`