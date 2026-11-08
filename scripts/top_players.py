"""Step 5: print the top 10 players by xG per 90 and by pressures per 90.

Reads the player_season_p90 view, so it only covers players with at least
600 minutes. Run from the project root with: python scripts/top_players.py
"""

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "scouting.db"
TOP_N = 10


def top_players(connection: sqlite3.Connection, metric: str, total: str) -> pd.DataFrame:
    """Return the top players by a per-90 column of player_season_p90."""
    query = f"""
        SELECT player_name, teams, ROUND(minutes) AS minutes, {total}, {metric}
        FROM player_season_p90
        ORDER BY {metric} DESC
        LIMIT {TOP_N}
    """
    table = pd.read_sql_query(query, connection)
    table.index = range(1, len(table) + 1)
    return table


def main() -> None:
    with sqlite3.connect(DB_PATH) as connection:
        by_xg = top_players(connection, "xg_p90", "xg")
        by_pressures = top_players(connection, "pressures_p90", "pressures")
    connection.close()

    print(f"Top {TOP_N} by xG per 90 (minimum 600 minutes)")
    print(by_xg.round(2).to_string())
    print(f"\nTop {TOP_N} by pressures per 90 (minimum 600 minutes)")
    print(by_pressures.round(2).to_string())


if __name__ == "__main__":
    main()
