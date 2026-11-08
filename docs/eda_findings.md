# Exploratory data analysis: findings

**The numbers in this document are a snapshot from the current `data/scouting.db`. The app computes them live.** Data: StatsBomb open data for the 2023/24 Frauen-Bundesliga (132 matches, 12 teams).

## 1. Data quality and treatment decisions

Minutes come from corrected lineup stints: each stint ends at the player's substitution or sending-off, and time off the pitch for treatment is not counted. All 16 metric totals in the database match the raw events.

| Check | Count | Treatment |
|---|---|---|
| Events with no location (9 administrative types, such as substitutions and half markers) | 4,187 | No metric reads their location, so no action |
| Passes and carries with no location | 0 | These coordinates feed the progressive and final-third metrics |
| Player-match rows with no position | 0 | None needed |
| Player-match rows with zero passes | 62 | Completion % stored as NULL, never 0 |
| Players whose season completion % differs from completed / attempted | 0 of 168 | Confirms the view uses season totals, not an average of match percentages |
| Players with more than one club | 5 | One season row, with minutes and counts summed across clubs |
| Goalkeepers excluded | 29 | No position group; their 274 rows stay in `player_match` so totals reconcile |
| Outfield players below 600 minutes excluded | 97 of 265 | Left out of season views and percentiles |
| Outlier flags | 81 | Kept (section 5) |

## 2. Minutes threshold

**First method and its flaw.** For each threshold T, the first method took players with at least T/2 minutes in each half of the season and correlated their per-90 values between halves. Mean reliability looked flat, from 0.70 at T = 300 to 0.75 at T = 1,200. But a threshold is only a floor: the median included player had 720 to 867 minutes per half, so every level measured players with about 1,440 to 1,730 minutes. The method could not show how reliable a metric is at T minutes.

**Minutes-matched redesign.** One fixed pool of 85 outfield players (600+ minutes in each half) is used at every level. For each T, whole matches are drawn in random order until each half reaches T/2 minutes. Within-group Spearman correlations between halves are averaged, weighted by group size, and the Spearman-Brown formula gives r_full. This is repeated over 200 draws.

| T | Actual minutes sampled (about) | Mean r_full | 10–90% of draws |
|---|---|---|---|
| 200 | 355 | 0.396 | 0.335–0.455 |
| 300 | 380 | 0.433 | 0.376–0.483 |
| 400 | 538 | 0.506 | 0.455–0.554 |
| 600 | 721 | 0.577 | 0.543–0.613 |
| 900 | 965 | 0.644 | 0.615–0.676 |
| 1,200 | 1,293 | 0.694 | 0.676–0.713 |

Reliability rises at every step and does not plateau.

**Rule set in advance.** Keep 600 if mean r_full at 600 is at least 0.70 and the gain from 600 to 900 is below 0.05. The rule failed on both conditions: mean r_full was 0.577 and the gain was 0.067. Interpolated on actual sampled minutes, reliability is 0.53 at 600 minutes and 0.63 at 900.

**Decision.** We kept 600 as a coverage/reliability trade-off, with reliability flags at metric level. No level reached 0.70, and a 900-minute cutoff would reach only 0.64 while dropping 32 of 168 players, including 8 of 23 strikers. This departs from the rule set in advance and is a judgment call.

## 3. Reliability tiers at T = 600

| Tier | Metric (r_full, 10–90% of draws) |
|---|---|
| High (≥ 0.70) | passes completed (0.83, 0.79–0.87); progressive passes (0.76, 0.71–0.81); pass completion % (0.75, 0.69–0.81); progressive carries (0.72, 0.65–0.78) |
| Moderate (0.55 to < 0.70) | pressures (0.69, 0.61–0.75); passes into final third (0.67, 0.60–0.74); aerial duels won (0.59, 0.51–0.68) |
| Low (< 0.55) | npxG (0.49, 0.37–0.61); ball recoveries (0.49, 0.36–0.61); shots (0.48, 0.34–0.60); successful dribbles (0.48, 0.36–0.59); non-penalty shots (0.48, 0.35–0.60); xG (0.48, 0.34–0.60); tackles won plus interceptions (0.48, 0.36–0.60); xA (0.45, 0.30–0.57); key passes (0.40, 0.26–0.53) |

Passing and progression metrics are the most repeatable. Shooting, chance creation and defensive actions should be treated as weak signals at this sample size.

## 4. Descriptive patterns by position

Median per 90, players with at least 600 minutes:

| Metric | Attacking Mid/Winger (36) | Center Back (37) | Def./Central Mid (41) | Full Back/Wing Back (31) | Striker (23) |
|---|---|---|---|---|---|
| npxG | 0.17 | 0.02 | 0.06 | 0.04 | 0.25 |
| Non-penalty shots | 1.75 | 0.34 | 1.02 | 0.66 | 1.86 |
| xA | 0.11 | 0.02 | 0.04 | 0.07 | 0.13 |
| Passes completed | 18.13 | 38.44 | 27.13 | 33.79 | 14.01 |
| Progressive passes | 3.22 | 6.60 | 5.42 | 7.86 | 1.82 |
| Progressive carries | 3.09 | 1.10 | 1.35 | 2.35 | 1.91 |
| Pressures | 18.24 | 10.45 | 18.84 | 16.10 | 21.37 |
| Tackles won + interceptions | 1.30 | 1.64 | 2.17 | 2.08 | 1.09 |
| Aerial duels won | 0.71 | 1.66 | 1.26 | 0.83 | 1.60 |

Positions differ enough that all rankings are made within position group. Strikers and attacking players lead shooting and chance creation, while full backs and center backs lead progressive passing.

## 5. Outliers

Within each group and metric, players outside 1.5 x IQR from the quartiles are flagged. There are 81 flags across 47 players (76 high, 5 low). Players with 600–900 minutes account for 5% of flags (4 flags) and 19% of the pool, so outliers are not low-sample noise; they are mostly regular starters. They are kept because they are real performances and percentiles are rank-based.

## 6. Correlations: pooled and within group

The matrix uses 14 metrics; xG and shots are left out as near-duplicates of npxG and non-penalty shots. Across the 91 pairs, mean absolute r is 0.33 pooled and 0.26 within group (the average weighted by group size), and pooled r is stronger for 66 pairs.

| Pair | Pooled r | Within-group r | Reading |
|---|---|---|---|
| Non-penalty shots, passes completed | -0.49 | 0.23 | The sign changes: strikers shoot more and pass less than center backs |
| Key passes, passes into final third | 0.02 | 0.49 | Pooling hides the link: different positions lead each metric |
| Pressures, passes completed | -0.65 | -0.41 | Within positions, players who press more complete fewer passes; team style is not controlled for |
| Progressive passes, passes into final third | 0.91 | 0.85 | Strong in both, so these two overlap |

## 7. Team level

The computed table passes every arithmetic check (22 matches per team, 2 or 3 points per match, goals and npxG balance), and it matches the [published final table](https://rsssf.org/tablesd/duit-wom2024.html) for all 12 teams.

Across 12 teams, npxG difference and points have Pearson r 0.94 and Spearman r 0.94; npxG difference and goal difference have 0.96 for both. Without the two highest and two lowest teams by npxG difference, the correlation with points is 0.71, so the extreme clubs drive much of the result. Relative to the fit of goal difference on npxG difference, SC Freiburg underperformed most (-18 against a fitted -2.5) and RB Leipzig overperformed most (-15 against a fitted -26.6).

## 8. Limitations

- **12 teams.** Team-level correlations are approximate and depend on a few clubs.
- **85-player stability pool of regular starters.** Reliability may differ for fringe players. The pool includes only 10 strikers.
- **Overshoot at low thresholds.** Whole matches exceed the target, so T = 200 and T = 300 sample nearly the same minutes (about 355 and 380). Tiers use nominal T, so "600" reflects about 721 actual minutes.
- **The 10–90% band reflects match sampling only.** It excludes uncertainty from the limited number of players.
- **23 strikers.** This is the smallest group at 600 minutes, so striker percentiles move in large steps.
