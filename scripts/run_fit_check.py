"""Check the tactical-fit scores against an independent pandas recomputation.

Run from the project root with: python scripts/run_fit_check.py
Exits with an error if any score, ranking, count or validation check fails.
"""

import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import fit  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
TOLERANCE = 1e-9
PERCENTILE_TOLERANCE = 1e-12
SHOW_TOP = 5

EXAMPLES = {
    "Progressive ball-playing center back": {
        "position_group": "Center Back",
        "weights": {
            "progressive_passes": 3,
            "passes_into_final_third": 2,
            "pass_completion_pct": 2,
            "progressive_carries": 1,
            "aerial_duels_won": 2,
        },
        "min_minutes": 900,
    },
    "Pressing striker": {
        "position_group": "Striker",
        "weights": {"pressures": 3, "npxg": 3, "tackles_interceptions": 1, "ball_recoveries": 1},
        "min_minutes": 1200,
    },
    "Creative winger": {
        "position_group": "Attacking Midfield/Winger",
        "weights": {"xa": 3, "key_passes": 2, "successful_dribbles": 2, "progressive_carries": 2},
    },
}

INVALID = {
    "unknown metric": {"position_group": "Striker", "weights": {"goals_scored": 1}},
    "bad group": {"position_group": "Forward", "weights": {"npxg": 1}},
    "min_minutes 300": {"position_group": "Striker", "weights": {"npxg": 1}, "min_minutes": 300},
    "excluded metric (xg)": {"position_group": "Striker", "weights": {"xg": 1}},
    "no weights": {"position_group": "Striker", "weights": {}},
    "seven metrics": {
        "position_group": "Striker",
        "weights": {name: 1 for name in fit.ALLOWED_METRICS[:7]},
    },
    "zero weight": {"position_group": "Striker", "weights": {"npxg": 0}},
    "boolean weight": {"position_group": "Striker", "weights": {"npxg": True}},
    "fractional min_minutes": {
        "position_group": "Striker",
        "weights": {"npxg": 1},
        "min_minutes": 600.5,
    },
    "unknown field": {"position_group": "Striker", "weights": {"npxg": 1}, "min_minute": 900},
    "not a mapping": ["Striker"],
}


def sqlite_percent_rank(values: pd.Series, higher_is_better: bool) -> pd.Series:
    """Return SQLite's PERCENT_RANK: (min-rank - 1) / (n - 1), and 0 for a single row."""
    ranks = values.rank(method="min", ascending=higher_is_better)
    return (ranks - 1) / (len(values) - 1) if len(values) > 1 else ranks * 0.0


def pandas_percentiles(connection: sqlite3.Connection) -> pd.DataFrame:
    """Return every allowed metric's percentile, recomputed in pandas within each group."""
    season = pd.read_sql_query("SELECT * FROM player_season_p90", connection)
    directions = dict(
        connection.execute("SELECT metric_name, higher_is_better FROM metric_definitions").fetchall()
    )
    for name in fit.ALLOWED_METRICS:
        column = fit.SEASON_COLUMNS[name]
        season[f"{column}_pctl"] = season.groupby("position_group")[column].transform(
            lambda values, better=bool(directions[name]): sqlite_percent_rank(values, better)
        )
    return season


def check_percentiles_match_view(connection: sqlite3.Connection, recomputed: pd.DataFrame) -> list[str]:
    """Return problems where the pandas percentiles differ from the view's."""
    view = pd.read_sql_query("SELECT * FROM player_percentiles", connection).set_index("player_id")
    ours = recomputed.set_index("player_id").loc[view.index]
    problems = []
    for name in fit.ALLOWED_METRICS:
        column = f"{fit.SEASON_COLUMNS[name]}_pctl"
        gap = float((view[column] - ours[column]).abs().max())
        if gap > PERCENTILE_TOLERANCE:
            problems.append(f"percentile {column}: max difference {gap:.3g}")
    return problems


def pandas_scores(recomputed: pd.DataFrame, requirement: dict) -> pd.DataFrame:
    """Return the full ranked list for a validated requirement, computed in pandas."""
    group = recomputed[recomputed["position_group"] == requirement["position_group"]]
    eligible = group[group["minutes"] >= requirement["min_minutes"]].copy()
    eligible["fit_score"] = 100 * sum(
        weight * eligible[f"{fit.SEASON_COLUMNS[name]}_pctl"]
        for name, weight in requirement["weights"].items()
    )
    return eligible.sort_values(
        ["fit_score", "minutes", "player_name"], ascending=[False, False, True], ignore_index=True
    )


def check_example(
    connection: sqlite3.Connection, recomputed: pd.DataFrame, label: str, req: dict
) -> list[str]:
    """Return problems for one example, comparing every eligible player SQL ranks."""
    requirement = fit.validate_requirement(req)
    expected = pandas_scores(recomputed, requirement)
    group_pool = int((recomputed["position_group"] == requirement["position_group"]).sum())
    result = fit.fit_scores(connection, req, top_n=max(len(expected), 1))
    table = result.table

    problems = []
    if result.pool_size != group_pool or result.eligible_players != len(expected):
        problems.append(
            f"{label}: counts {result.pool_size}/{result.eligible_players}, "
            f"expected {group_pool}/{len(expected)}"
        )
    if abs(sum(result.weights.values()) - 1) > TOLERANCE:
        problems.append(f"{label}: weights sum to {sum(result.weights.values())}")
    if table["player"].tolist() != expected["player_name"].tolist():
        problems.append(f"{label}: SQL and pandas order players differently")
        return problems
    if table["rank"].tolist() != list(range(1, len(table) + 1)):
        problems.append(f"{label}: ranks are not 1..n")
    score_gap = float((table["fit_score"] - expected["fit_score"]).abs().max())
    if score_gap > TOLERANCE:
        problems.append(f"{label}: fit_score max difference {score_gap:.3g}")
    for name in requirement["weights"]:
        column = fit.SEASON_COLUMNS[name]
        for output in (column, f"{column}_pctl"):
            gap = float((table[output] - expected[output]).abs().max())
            if gap > TOLERANCE:
                problems.append(f"{label}: {output} max difference {gap:.3g}")
    return problems


def check_invalid_requirements() -> list[str]:
    """Return the invalid requirements that did not raise RequirementError."""
    problems = []
    for label, req in INVALID.items():
        try:
            fit.validate_requirement(req)
        except fit.RequirementError as error:
            print(f"  rejected as expected - {label}: {error}")
        else:
            problems.append(f"invalid requirement accepted: {label}")
    return problems


def print_shortlist(result: fit.FitResult, label: str) -> None:
    """Print the top of a shortlist with what was scored and both counts."""
    weights = ", ".join(f"{name} {weight:.3f}" for name, weight in result.weights.items())
    print(f"\n=== {label} ===")
    print(f"Group: {result.position_group}; min_minutes: {result.min_minutes}")
    print(f"Normalized weights: {weights}")
    print(
        f"Reference pool (600+ minutes): {result.pool_size}; "
        f"eligible at {result.min_minutes}+: {result.eligible_players}"
    )
    print(result.table.head(SHOW_TOP).round(3).to_string(index=False))


def main() -> None:
    pd.set_option("display.width", 250, "display.max_columns", None)
    problems = []
    with sqlite3.connect(DB_PATH) as connection:
        defined = {row[0] for row in connection.execute("SELECT metric_name FROM metric_definitions")}
        expected_allowed = defined - set(fit.EXCLUDED_METRICS)
        if set(fit.ALLOWED_METRICS) != expected_allowed:
            problems.append("ALLOWED_METRICS does not match metric_definitions minus xg and shots")

        recomputed = pandas_percentiles(connection)
        problems += check_percentiles_match_view(connection, recomputed)
        for label, req in EXAMPLES.items():
            problems += check_example(connection, recomputed, label, req)
            print_shortlist(fit.fit_scores(connection, req), label)
    connection.close()

    print("\n=== Invalid requirements ===")
    problems += check_invalid_requirements()

    if problems:
        print("\nFAILED:")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print(
        f"\nAll checks passed: {len(fit.ALLOWED_METRICS)} percentile columns match the view, "
        f"{len(EXAMPLES)} examples match pandas to {TOLERANCE:g}, "
        f"{len(INVALID)} invalid requirements rejected."
    )


if __name__ == "__main__":
    main()
