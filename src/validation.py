"""Check the finished database against the raw events and print review tables.

raw_event_totals deliberately shares no code with src/metrics.py. It walks the
events one at a time and, where StatsBomb records a fact twice, reads the other
record: key passes and xA come from the shots rather than the passes.
"""

import math
import sqlite3
from collections import Counter

import pandas as pd

from src.metrics import METRICS

TOLERANCE = 1e-6
REVIEW_METRICS = ["npxg_p90", "xa_p90", "progressive_carries_p90", "tackles_interceptions_p90"]
TOP_N = 5


def _is_text(value) -> bool:
    """Return True when an optional string field holds a value."""
    return isinstance(value, str)


def _gain_in_tenths(start_x: float, end_x: float) -> int:
    """Return forward movement in whole tenths of a yard, avoiding float error."""
    return round(end_x * 10) - round(start_x * 10)


def raw_event_totals(events: pd.DataFrame) -> Counter:
    """Recount every metric's season total directly from the raw events."""
    totals: Counter = Counter()
    for event in events.itertuples(index=False):
        if not pd.notna(event.player_id):
            continue
        kind = event.type
        if kind == "Shot":
            xg = float(event.shot_statsbomb_xg)
            totals["shots"] += 1
            totals["xg"] += xg
            if event.shot_type != "Penalty":
                totals["np_shots"] += 1
                totals["npxg"] += xg
            if _is_text(event.shot_key_pass_id):
                totals["key_passes"] += 1
                totals["xa"] += xg
            if event.shot_aerial_won == True:  # noqa: E712
                totals["aerial_duels_won"] += 1
        elif kind == "Pass":
            totals["passes_attempted"] += 1
            if event.pass_aerial_won == True:  # noqa: E712
                totals["aerial_duels_won"] += 1
            if _is_text(event.pass_outcome):
                continue
            totals["passes_completed"] += 1
            start, end = event.location[0], event.pass_end_location[0]
            if start >= 40 and _gain_in_tenths(start, end) >= 100:
                totals["progressive_passes"] += 1
            if start < 80 < end:
                totals["passes_into_final_third"] += 1
        elif kind == "Carry":
            start = event.location[0]
            if start >= 40 and _gain_in_tenths(start, event.carry_end_location[0]) >= 100:
                totals["progressive_carries"] += 1
        elif kind == "Dribble":
            totals["successful_dribbles"] += event.dribble_outcome == "Complete"
        elif kind == "Pressure":
            totals["pressures"] += 1
        elif kind == "Interception":
            if event.interception_outcome in ("Won", "Success In Play", "Success Out"):
                totals["tackles_interceptions"] += 1
        elif kind == "Duel":
            if event.duel_type == "Tackle" and event.duel_outcome in (
                "Won",
                "Success In Play",
                "Success Out",
            ):
                totals["tackles_interceptions"] += 1
        elif kind == "Ball Recovery":
            totals["ball_recoveries"] += event.ball_recovery_recovery_failure != True  # noqa: E712
        elif kind == "Clearance":
            totals["aerial_duels_won"] += event.clearance_aerial_won == True  # noqa: E712
        elif kind == "Miscontrol":
            totals["aerial_duels_won"] += event.miscontrol_aerial_won == True  # noqa: E712
    return totals


def print_reconciliation(connection: sqlite3.Connection, events: pd.DataFrame) -> bool:
    """Print database totals beside raw totals and return True if all of them match."""
    raw = raw_event_totals(events)
    names = [m.name for m in METRICS if m.name != "pass_completion_pct"]
    names.insert(names.index("passes_completed"), "passes_attempted")

    sums = ", ".join(f"SUM({name})" for name in names)
    database = dict(zip(names, connection.execute(f"SELECT {sums} FROM player_match").fetchone()))

    print("Metric totals: player_match against raw events")
    all_match = True
    for name in names:
        matches = math.isclose(database[name], raw[name], abs_tol=TOLERANCE)
        all_match = all_match and matches
        status = "OK" if matches else "MISMATCH"
        print(f"  {name:<26}{database[name]:>13,.2f}{raw[name]:>13,.2f}  {status}")

    completion = 100 * raw["passes_completed"] / raw["passes_attempted"]
    print(f"  League pass completion: {completion:.1f}%")
    return all_match


def print_view_checks(connection: sqlite3.Connection) -> bool:
    """Print structural checks on the views and return True if all of them pass."""
    checks = {
        "Foreign key violations": "SELECT COUNT(*) FROM pragma_foreign_key_check",
        "Season rows under 600 minutes": "SELECT COUNT(*) FROM player_season_p90 WHERE minutes < 600",
        "Season rows with no position group": (
            "SELECT COUNT(*) FROM player_season_p90 WHERE position_group IS NULL"
        ),
        "Season rows minus percentile rows": (
            "SELECT (SELECT COUNT(*) FROM player_season_p90)"
            " - (SELECT COUNT(*) FROM player_percentiles)"
        ),
    }
    print("\nStructural checks (each should be 0)")
    all_pass = True
    for label, query in checks.items():
        value = connection.execute(query).fetchone()[0]
        all_pass = all_pass and value == 0
        print(f"  {label}: {value}")
    return all_pass


def print_position_counts(connection: sqlite3.Connection) -> None:
    """Print how many players fall in each position group."""
    query = """
        SELECT
            COALESCE(p.position_group, '(Goalkeeper, no group)') AS position_group,
            COUNT(*) AS played,
            SUM(p.player_id IN (SELECT player_id FROM season_minutes)) AS with_600_minutes
        FROM players AS p
        WHERE p.primary_position IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """
    connection.execute(
        """
        CREATE TEMP VIEW season_minutes AS
        SELECT player_id FROM player_match GROUP BY player_id HAVING SUM(minutes) >= 600
        """
    )
    print("\nPlayers per position group")
    print(pd.read_sql_query(query, connection).to_string(index=False))


def print_top_players(connection: sqlite3.Connection) -> None:
    """Print the top players per position group for the review metrics."""
    for metric in REVIEW_METRICS:
        query = f"""
            SELECT position_group, player_name, teams, ROUND(minutes) AS minutes,
                   ROUND({metric}, 2) AS {metric}
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY position_group ORDER BY {metric} DESC
                ) AS position_rank
                FROM player_season_p90
            )
            WHERE position_rank <= {TOP_N}
            ORDER BY position_group, position_rank
        """
        print(f"\nTop {TOP_N} per position group by {metric}")
        print(pd.read_sql_query(query, connection).to_string(index=False))


def run_validation(connection: sqlite3.Connection, events: pd.DataFrame) -> bool:
    """Print every validation table and return True if all automated checks pass."""
    totals_match = print_reconciliation(connection, events)
    views_pass = print_view_checks(connection)
    print_position_counts(connection)
    print_top_players(connection)
    return totals_match and views_pass
