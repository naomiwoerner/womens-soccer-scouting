"""Build the SQLite scouting database from the raw Parquet files.

Rebuilds data/scouting.db from scratch each time it runs, then validates it.
Run from the project root with: python scripts/build_db.py
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import EVENTS_PATH, LINEUPS_PATH, MATCHES_PATH  # noqa: E402
from src.metrics import EVENT_COLUMNS, EVENT_TYPES, METRICS, build_player_match  # noqa: E402
from src.minutes import compute_minutes  # noqa: E402
from src.positions import position_minutes, primary_positions  # noqa: E402
from src.validation import run_validation  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
MIN_SEASON_MINUTES = 600
LOCATION_METRIC_EVENT_TYPES = ("Pass", "Carry")


def player_match_columns_sql() -> str:
    """Return the metric column definitions for the player_match table."""
    columns = [
        f"    {m.name} {m.column_type}" + ("" if m.name == "pass_completion_pct" else " NOT NULL")
        for m in METRICS
    ]
    columns.append("    passes_attempted INTEGER NOT NULL")
    return ",\n".join(columns)


def season_columns_sql() -> str:
    """Return the season total and rate columns for player_season_p90."""
    lines = []
    for metric in METRICS:
        if metric.per_90:
            lines.append(f"    SUM(pm.{metric.name}) AS {metric.name}")
            lines.append(f"    SUM(pm.{metric.name}) * 90.0 / SUM(pm.minutes) AS {metric.season_column}")
        else:
            # A percentage of the season's passes, not an average of match percentages.
            lines.append(
                "    SUM(pm.passes_completed) * 100.0 / NULLIF(SUM(pm.passes_attempted), 0)"
                f" AS {metric.season_column}"
            )
    return ",\n".join(lines)


def percentile_columns_sql() -> str:
    """Return one PERCENT_RANK column per metric for player_percentiles."""
    lines = []
    for metric in METRICS:
        direction = "ASC" if metric.higher_is_better else "DESC"
        lines.append(
            f"    PERCENT_RANK() OVER (PARTITION BY position_group "
            f"ORDER BY {metric.season_column} {direction}) AS {metric.season_column}_pctl"
        )
    return ",\n".join(lines)


def schema_sql() -> str:
    """Return the full database schema."""
    return f"""
CREATE TABLE players (
    player_id        INTEGER PRIMARY KEY,
    player_name      TEXT NOT NULL,
    player_nickname  TEXT,
    country          TEXT,
    primary_position TEXT,
    position_group   TEXT
);

CREATE TABLE teams (
    team_id   INTEGER PRIMARY KEY,
    team_name TEXT NOT NULL
);

CREATE TABLE player_positions (
    player_id        INTEGER NOT NULL REFERENCES players (player_id),
    team_id          INTEGER NOT NULL REFERENCES teams (team_id),
    primary_position TEXT NOT NULL,
    position_group   TEXT NOT NULL,
    PRIMARY KEY (player_id, team_id)
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
{player_match_columns_sql()},
    PRIMARY KEY (player_id, match_id)
);

CREATE TABLE metric_definitions (
    metric_name      TEXT PRIMARY KEY,
    definition       TEXT NOT NULL,
    higher_is_better INTEGER NOT NULL
);

CREATE TABLE data_quality (
    check_name TEXT PRIMARY KEY,
    count      INTEGER NOT NULL,
    treatment  TEXT NOT NULL
);

CREATE VIEW player_season_p90 AS
SELECT
    p.player_id,
    p.player_name,
    p.position_group,
    GROUP_CONCAT(DISTINCT t.team_name) AS teams,
    COUNT(*) AS matches,
    SUM(pm.minutes) AS minutes,
{season_columns_sql()}
FROM player_match AS pm
JOIN players AS p ON p.player_id = pm.player_id
JOIN teams AS t ON t.team_id = pm.team_id
WHERE p.position_group IS NOT NULL
GROUP BY p.player_id, p.player_name, p.position_group
HAVING SUM(pm.minutes) >= {MIN_SEASON_MINUTES};

CREATE VIEW player_percentiles AS
SELECT
    player_id,
    player_name,
    position_group,
    teams,
    minutes,
{percentile_columns_sql()}
FROM player_season_p90;
"""


def build_teams(matches: pd.DataFrame) -> pd.DataFrame:
    """Return one row per team from the home and away columns."""
    home = matches[["home_team_id", "home_team"]].set_axis(["team_id", "team_name"], axis=1)
    away = matches[["away_team_id", "away_team"]].set_axis(["team_id", "team_name"], axis=1)
    return pd.concat([home, away]).drop_duplicates("team_id").sort_values("team_id")


def build_players(lineups: pd.DataFrame, by_position: pd.DataFrame) -> pd.DataFrame:
    """Return one row per squad player, with their season position group.

    Players who never appeared, and goalkeepers, have no position group.
    """
    columns = ["player_id", "player_name", "player_nickname", "country"]
    players = lineups[columns].drop_duplicates("player_id").sort_values("player_id")
    return players.merge(
        primary_positions(by_position, ["player_id"]),
        on="player_id",
        how="left",
        validate="one_to_one",
    )


def build_player_positions(by_position: pd.DataFrame) -> pd.DataFrame:
    """Return each outfield player's position group at each of their clubs."""
    positions = primary_positions(by_position, ["player_id", "team_id"])
    return positions[positions["position_group"].notna()]


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


def build_metric_definitions() -> pd.DataFrame:
    """Return one row per metric, describing how it is calculated."""
    return pd.DataFrame(
        {
            "metric_name": [m.name for m in METRICS],
            "definition": [m.definition for m in METRICS],
            "higher_is_better": [int(m.higher_is_better) for m in METRICS],
        }
    )


def build_data_quality() -> pd.DataFrame:
    """Return the data-quality checks that need the raw events.

    data/raw is not in the repository, so the app reads these results from the
    database. Passes and carries are always listed, even at zero, because the
    progressive and final-third metrics depend on their coordinates.
    """
    events = pd.read_parquet(EVENTS_PATH, columns=["type", "location"])
    missing = events.loc[events["location"].isna(), "type"].value_counts().to_dict()
    for event_type in LOCATION_METRIC_EVENT_TYPES:
        missing.setdefault(event_type, 0)

    rows = []
    for event_type, count in sorted(missing.items()):
        if event_type in LOCATION_METRIC_EVENT_TYPES:
            treatment = (
                "Used by the progressive and final-third metrics, which need both coordinates."
                if count == 0
                else "Used by location metrics: these events break the coordinate calculations."
            )
        else:
            treatment = "Event has no on-pitch position. No metric reads its location, so no action."
        rows.append(
            {
                "check_name": f"Events missing location: {event_type}",
                "count": int(count),
                "treatment": treatment,
            }
        )
    return pd.DataFrame(rows)


def write_database(tables: dict[str, pd.DataFrame]) -> None:
    """Replace the database file with one containing the given tables."""
    DB_PATH.unlink(missing_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(schema_sql())
        for name, frame in tables.items():
            frame.to_sql(name, connection, if_exists="append", index=False)
    connection.close()


def main() -> None:
    matches = pd.read_parquet(MATCHES_PATH)
    lineups = pd.read_parquet(LINEUPS_PATH)
    events = pd.read_parquet(
        EVENTS_PATH, columns=EVENT_COLUMNS, filters=[("type", "in", EVENT_TYPES)]
    )

    minutes = compute_minutes(lineups, events)
    by_position = position_minutes(lineups, events)
    write_database(
        {
            "teams": build_teams(matches),
            "players": build_players(lineups, by_position),
            "player_positions": build_player_positions(by_position),
            "matches": build_matches(matches, minutes),
            "player_match": build_player_match(minutes, events),
            "metric_definitions": build_metric_definitions(),
            "data_quality": build_data_quality(),
        }
    )

    print(f"Wrote {DB_PATH}")
    with sqlite3.connect(DB_PATH) as connection:
        for name in (
            "players",
            "player_positions",
            "teams",
            "matches",
            "player_match",
            "metric_definitions",
            "data_quality",
            "player_season_p90",
            "player_percentiles",
        ):
            count = connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
            print(f"  {name}: {count:,} rows")
        print()
        passed = run_validation(connection, events)
    connection.close()

    if not passed:
        sys.exit("Validation failed: see the MISMATCH or non-zero lines above.")
    print("\nAll automated checks passed.")


if __name__ == "__main__":
    main()
