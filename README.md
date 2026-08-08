# BL Tipper
ML-Modell zur Vorhersage der Bundesliga-Saison 26/27

## Status
Working on getting data cleaned up.

### To-Do: Feature Engineering

- [ ] Longformat bauen (ein Team pro Zeile, Home- und Away-Perspektive getrennt)
- [ ] Pro Team zeitlich nach `Date` sortieren
- [ ] Rolling-Features berechnen (mit `shift(1)` + `rolling(window=n)`)
  - [ ] Form (Punkte letzte 5 Spiele)
  - [ ] Ø Tore geschossen / kassiert
  - [ ] Ø Shots / Corners / Fouls / Cards (D1 durchgängig, D2 erst ab 2017)
- [ ] Tabellenplatz-Feature (kumulative Punkte vor jedem Spieltag, pro Saison)
- [ ] Zurück ins Wide-Format joinen (Home-Rolling-Stats + Away-Rolling-Stats pro Spiel)
- [ ] D1 + D2 zusammenführen (mit `Div` als Kontext-Feature)
- [ ] Finaler NaN-Check (erste Spiele je Saison ohne Rolling-History droppen/handhaben)
- [ ] Speichern als `data/processed/features.csv`