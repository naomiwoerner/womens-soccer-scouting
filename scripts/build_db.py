"""Step 4: build the SQLite scouting database from the raw Parquet files.

Rebuilds data/scouting.db from scratch each time it runs.
Run from the project root with: python scripts/build_db.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import EVENTS_PATH, LINEUPS_PATH, MATCHES_PATH  # noqa: E402
from src.minutes import compute_minutes  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
MIN_SEASON_MINUTES = 600

SCHEMA = f"""
CREATE TABLE players (
    player_id       INTEGER PRIMARY KEY,
    player_name     TEXT NOT NULL,
    player_nickname TEXT,
    country         TEXT
);

CREATE TABLE teams (
    team_id   INTEGER PRIMARY KEY,
    team_name TEXT NOT NULL
);

CREATE TABLE matches (
    match_id      INTEGER PRIMARY KEY,
    match_date    TEXT NOT NULL,
    match_week    INTEGER NOT NULL,
    home_team_id  INTEGER NOT NULL REFERENCES teams (team_id),
    away_team_id  INTEGER NOT NULL REFERENCES teams (team_id),
    home_score    INTEGER NOT NULL,
    away_score    INTEGER NOT NULL,
    match_minutes REAL NOT NULL
);

CREATE TABLE player_match (
    player_id INTEGER NOT NULL REFERENCES players (player_id),
    match_id  INTEGER NOT NULL REFERENCES matches (match_id),
    team_id   INTEGER NOT NULL REFERENCES teams (team_id),
    minutes   REAL NOT NULL,
    shots     INTEGER NOT NULL,
    xg        REAL NOT NULL,
    pressures INTEGER NOT NULL,
    PRIMARY KEY (player_id, match_id)
);

CREATE VIEW player_season_p90 AS
SELECT
    p.player_id,
    p.player_name,
    GROUP_CONCAT(DISTINCT t.team_name) AS teams,
    COUNT(*) AS matches,
    SUM(pm.minutes) AS minutes,
    SUM(pm.shots) AS shots,
    SUM(pm.xg) AS xg,
    SUM(pm.pressures) AS pressures,
    SUM(pm.shots) * 90.0 / SUM(pm.minutes) AS shots_p90,
    SUM(pm.xg) * 90.0 / SUM(pm.minutes) AS xg_p90,
    SUM(pm.pressures) * 90.0 / SUM(pm.minutes) AS pressures_p90
FROM player_match AS pm
JOIN players AS p ON p.player_id = pm.player_id
JOIN teams AS t ON t.team_id = pm.team_id
GROUP BY p.player_id, p.player_name
HAVING SUM(pm.minutes) >= {MIN_SEASON_MINUTES};
"""


def build_teams(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one row per team from the home and away columns."""
    home = matches[["home_team_id", "home_team"]].set_axis(["team_id", "team_name"], axis=1)
    away = matches[["away_team_id", "away_team"]].set_axis(["team_id", "team_name"], axis=1)
    return pd.concat([home, away]).drop_duplicates("team_id").sort_values("team_id")


def build_players(lineups: pd.DataFrame) -> pd.DataFrame:
    """Return one row per player named in any squad."""
    columns = ["player_id", "player_name", "player_nickname", "country"]
    return lineups[columns].drop_duplicates("player_id").sort_values("player_id")


def build_matches(matches: pd.DataFrame, minutes: pd.DataFrame) -> pd.DataFrame:
    """Return the match table, including each match's actual length."""
    lengths = minutes.groupby("match_id", as_index=False)["match_minutes"].first()
    columns = [
        "match_id",
        "match_date",
        "match_week",
        "home_team_id",
        "away_team_id",
        "home_score",
        "away_score",
    ]
    return matches[columns].merge(lengths, on="match_id", validate="one_to_one")


def build_player_match(minutes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return per-player, per-match minutes with shots, xG and pressures."""
    is_shot = events["type"] == "Shot"
    stats = (
        events.assign(
            shots=is_shot.astype(int),
            xg=events["shot_statsbomb_xg"].where(is_shot, 0.0),
            pressures=(events["type"] == "Pressure").astype(int),
        )
        .groupby(["match_id", "player_id"], as_index=False)[["shots", "xg", "pressures"]]
        .sum()
    )
    stats["player_id"] = stats["player_id"].astype(int)

    columns = ["player_id", "match_id", "team_id", "minutes"]
    player_match = minutes[columns].merge(
        stats, on=["match_id", "player_id"], how="left", validate="one_to_one"
    )
    return player_match.fillna({"shots": 0, "xg": 0.0, "pressures": 0}).astype(
        {"shots": int, "pressures": int}
    )


def write_database(tables: dict[str, pd.DataFrame]) -> None:
    """Replace the database file with one containing the given tables."""
    DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(SCHEMA)
        for name, frame in tables.items():
            frame.to_sql(name, connection, if_exists="append", index=False)
    connection.close()


def main() -> None:
    matches = pd.read_parquet(MATCHES_PATH)
    lineups = pd.read_parquet(LINEUPS_PATH)
    events = pd.read_parquet(
        EVENTS_PATH,
        columns=[
            "match_id",
            "type",
            "period",
            "minute",
            "second",
            "player_id",
            "shot_statsbomb_xg",
        ],
        filters=[("type", "in", ["Half End", "Substitution", "Shot", "Pressure"])],
    )

    minutes = compute_minutes(lineups, events)
    player_match = build_player_match(minutes, events)
    write_database(
        {
            "teams": build_teams(matches),
            "players": build_players(lineups),
            "matches": build_matches(matches, minutes),
            "player_match": player_match,
        }
    )

    # Every shot and pressure in the raw events should reach player_match.
    raw_shots = events[events["type"] == "Shot"]
    print(f"Wrote {DB_PATH}")
    print(f"  Shots: {int(player_match['shots'].sum()):,} in database, {len(raw_shots):,} raw")
    print(
        f"  xG: {player_match['xg'].sum():.2f} in database, "
        f"{raw_shots['shot_statsbomb_xg'].sum():.2f} raw"
    )
    print(
        f"  Pressures: {int(player_match['pressures'].sum()):,} in database, "
        f"{(events['type'] == 'Pressure').sum():,} raw"
    )
    with sqlite3.connect(DB_PATH) as connection:
        for name in ("players", "teams", "matches", "player_match", "player_season_p90"):
            count = connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"  {name}: {count:,} rows")
    connection.close()


if __name__ == "__main__":
    main()
