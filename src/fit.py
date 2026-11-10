"""Deterministic tactical-fit scores for a scout's requirement.

A requirement names a position group, up to six weighted metrics and a minutes
minimum. Each eligible player's fit score is 100 x the weighted sum of their
percentiles within the group. Functions that read data take an open SQLite
connection; nothing here prints or imports Streamlit.
"""

import math
import numbers
import sqlite3
from dataclasses import dataclass

import pandas as pd

from src.metrics import METRICS

# xG and shots include penalties and nearly duplicate npxG and non-penalty shots.
REPLACEMENTS = {"xg": "npxg", "shots": "np_shots"}
EXCLUDED_METRICS = tuple(REPLACEMENTS)
ALLOWED_METRICS = tuple(m.name for m in METRICS if m.name not in EXCLUDED_METRICS)
SEASON_COLUMNS = {m.name: m.season_column for m in METRICS}
OUTFIELD_GROUPS = (
    "Attacking Midfield/Winger",
    "Center Back",
    "Defensive/Central Midfield",
    "Full Back/Wing Back",
    "Striker",
)
REQUIREMENT_KEYS = ("position_group", "weights", "min_minutes")
MAX_WEIGHTED_METRICS = 6
DEFAULT_MIN_MINUTES = 600
MAX_MIN_MINUTES = 1200

# Display wording shared by the app and the scouting report.
EXCELLENT_FIT, STRONG_FIT, MODERATE_FIT = 85.0, 70.0, 55.0
FIT_LABELS = (
    (EXCELLENT_FIT, "Excellent fit"),
    (STRONG_FIT, "Strong fit"),
    (MODERATE_FIT, "Moderate fit"),
)
LIMITED_FIT_LABEL = "Limited fit"
FIT_SCORE_DECIMALS = 1
GROUP_LABELS = {
    "Attacking Midfield/Winger": "attacking midfielders and wingers",
    "Center Back": "center backs",
    "Defensive/Central Midfield": "defensive and central midfielders",
    "Full Back/Wing Back": "full backs and wing backs",
    "Striker": "strikers",
}
METRIC_NAMES = {
    "np_shots": "non-penalty shots",
    "npxg": "non-penalty expected goals (npxG)",
    "xa": "expected assists (xA)",
    "key_passes": "key passes",
    "passes_completed": "completed passes",
    "pass_completion_pct": "pass completion",
    "progressive_passes": "progressive passes",
    "passes_into_final_third": "passes into the final third",
    "progressive_carries": "progressive carries",
    "successful_dribbles": "successful dribbles",
    "pressures": "pressures",
    "tackles_interceptions": "tackles won and interceptions",
    "ball_recoveries": "ball recoveries",
    "aerial_duels_won": "aerial duels won",
}
assert set(METRIC_NAMES) == set(ALLOWED_METRICS), "METRIC_NAMES must name every allowed metric"
assert set(GROUP_LABELS) == set(OUTFIELD_GROUPS), "GROUP_LABELS must name every group"
# PERCENT_RANK x (n - 1) is a whole number in exact arithmetic; this absorbs float error.
RANK_TOLERANCE = 1e-6


class RequirementError(ValueError):
    """Raised when a requirement is invalid. The message is written for the user."""


@dataclass(frozen=True)
class FitResult:
    """A fit-score shortlist and what was scored.

    table has rank, player_id, player, team, minutes, fit_score and, for each weighted
    metric, its percentile and per-90 value. weights are the normalized weights
    that were scored. pool_size is the group's reference pool (players with at
    least 600 minutes, the pool percentiles come from); eligible_players is how
    many of them meet min_minutes, before top_n is applied.
    """

    table: pd.DataFrame
    position_group: str
    weights: dict[str, float]
    min_minutes: int
    pool_size: int
    eligible_players: int


def _is_number(value) -> bool:
    """Return True for a real, finite number that is not a boolean."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _check_position_group(group, errors: list[str]) -> None:
    """Add an error if group is not exactly one of the outfield groups."""
    if group not in OUTFIELD_GROUPS:
        errors.append(
            f"position_group must be one of: {', '.join(OUTFIELD_GROUPS)} (got {group!r})."
        )


def _check_weights(weights, errors: list[str]) -> dict[str, float]:
    """Add errors for invalid weights and return them normalized to sum to 1."""
    if not isinstance(weights, dict):
        errors.append("weights must be a mapping of metric name to a positive number.")
        return {}
    if not 1 <= len(weights) <= MAX_WEIGHTED_METRICS:
        errors.append(f"weights must name 1 to {MAX_WEIGHTED_METRICS} metrics (got {len(weights)}).")
    excluded = [name for name in weights if name in EXCLUDED_METRICS]
    if excluded:
        errors.append(
            " ".join(
                f"{name!r} is not used for scoring; use {REPLACEMENTS[name]!r} instead."
                for name in excluded
            )
        )
    unknown = [name for name in weights if name not in ALLOWED_METRICS and name not in excluded]
    if unknown:
        errors.append(
            f"Unknown metric(s): {', '.join(map(repr, unknown))}. "
            f"Allowed metrics: {', '.join(ALLOWED_METRICS)}."
        )
    invalid = [name for name, value in weights.items() if not _is_number(value) or value <= 0]
    if invalid:
        errors.append(
            f"Each weight must be a number greater than 0; check: {', '.join(map(repr, invalid))}."
        )
    if not weights or excluded or unknown or invalid:
        return {}
    total = float(sum(weights.values()))
    return {name: float(value) / total for name, value in weights.items()}


def _check_min_minutes(min_minutes, errors: list[str]) -> int:
    """Add an error if min_minutes is not a whole number from 600 to 1200."""
    if not _is_number(min_minutes) or float(min_minutes) != int(min_minutes):
        errors.append(
            f"min_minutes must be a whole number from {DEFAULT_MIN_MINUTES} to "
            f"{MAX_MIN_MINUTES} (got {min_minutes!r})."
        )
        return DEFAULT_MIN_MINUTES
    if not DEFAULT_MIN_MINUTES <= min_minutes <= MAX_MIN_MINUTES:
        errors.append(
            f"min_minutes must be from {DEFAULT_MIN_MINUTES} to {MAX_MIN_MINUTES} "
            f"(got {min_minutes:g})."
        )
    return int(min_minutes)


def validate_requirement(req: dict) -> dict:
    """Return a checked requirement with weights normalized to sum to 1.

    Expected shape: {"position_group": str, "weights": {metric: number},
    "min_minutes": int}. min_minutes defaults to 600 when missing or None.
    Every problem is collected and raised together as one RequirementError, so
    a user (or an LLM that wrote the requirement) can fix them all at once.
    """
    if not isinstance(req, dict):
        raise RequirementError("The requirement must be a mapping with position_group and weights.")
    errors: list[str] = []
    unknown_keys = [key for key in req if key not in REQUIREMENT_KEYS]
    if unknown_keys:
        errors.append(
            f"Unknown field(s): {', '.join(map(repr, unknown_keys))}. "
            f"Allowed fields: {', '.join(REQUIREMENT_KEYS)}."
        )
    missing = [key for key in ("position_group", "weights") if key not in req]
    if missing:
        errors.append(f"Missing required field(s): {', '.join(missing)}.")

    _check_position_group(req.get("position_group"), errors)
    weights = _check_weights(req.get("weights"), errors)
    min_minutes = req.get("min_minutes")
    min_minutes = _check_min_minutes(
        DEFAULT_MIN_MINUTES if min_minutes is None else min_minutes, errors
    )
    if errors:
        raise RequirementError(" ".join(errors))
    return {"position_group": req["position_group"], "weights": weights, "min_minutes": min_minutes}


def fit_label(score: float) -> str:
    """Return the wording for a fit score: Excellent, Strong, Moderate or Limited fit.

    The score is rounded to one decimal first, so the label always agrees with
    the score as displayed (84.96 shows as 85.0 and is an Excellent fit).
    """
    shown = round(float(score), FIT_SCORE_DECIMALS)
    for cutoff, label in FIT_LABELS:
        if shown >= cutoff:
            return label
    return LIMITED_FIT_LABEL


def players_ranked_below(percentile: float, pool_size: int) -> int:
    """Return how many players in the pool rank strictly below this percentile.

    PERCENT_RANK = (rank - 1) / (n - 1), and rank - 1 is the number of players
    with a strictly lower value, so percentile x (n - 1) is that count.
    """
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size < 1:
        raise ValueError("pool_size must be a positive whole number")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0 and 1")
    below = percentile * (pool_size - 1)
    count = round(below)
    if abs(below - count) > RANK_TOLERANCE:
        raise ValueError(f"percentile {percentile} is not a PERCENT_RANK value for a pool of {pool_size}")
    return count


def rank_phrase(percentile: float, pool_size: int, group_label: str) -> str:
    """Return plain wording for a percentile, such as "better than 15 of 22 other strikers".

    The count comes exactly from PERCENT_RANK (see players_ranked_below). Players
    tied with her are not counted as below her, so "best of 23" means no one
    is level or above. "lowest of 23" means no one ranks below her, which
    includes being tied for lowest.
    """
    below = players_ranked_below(percentile, pool_size)
    others = pool_size - 1
    if pool_size == 1:
        return f"the only one of the {group_label} in the pool"
    if below == others:
        return f"best of {pool_size} {group_label}"
    if below == 0:
        return f"lowest of {pool_size} {group_label}"
    return f"better than {below} of {others} other {group_label}"


def _check_database_matches(conn: sqlite3.Connection) -> None:
    """Raise if the database's metrics or groups no longer match this module.

    ALLOWED_METRICS and OUTFIELD_GROUPS are defined in code so a requirement can
    be validated without a database, so this guards against the two drifting apart.
    """
    defined = {row[0] for row in conn.execute("SELECT metric_name FROM metric_definitions")}
    missing_metrics = sorted(set(ALLOWED_METRICS) - defined)
    groups = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT position_group FROM players WHERE position_group IS NOT NULL"
        )
    }
    if missing_metrics or groups != set(OUTFIELD_GROUPS):
        raise RuntimeError(
            "The database does not match src/fit.py: "
            f"metrics missing from metric_definitions {missing_metrics}, "
            f"database groups {sorted(groups)}."
        )


def fit_scores(conn: sqlite3.Connection, req: dict, top_n: int = 10) -> FitResult:
    """Return the top players for a requirement, ranked by fit score.

    fit_score = 100 x sum(weight x percentile), with weights normalized to sum
    to 1, so it runs from 0 to 100. Percentiles always come from the group's
    full pool of players with at least 600 minutes (the player_percentiles
    view); min_minutes only filters who appears on the shortlist, so a player's
    percentile does not change with the minutes setting. Ties are broken by
    minutes (more first), then player name.
    """
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive whole number")
    requirement = validate_requirement(req)
    _check_database_matches(conn)
    weights = requirement["weights"]

    score_terms, metric_columns, weight_params = [], [], []
    for name, weight in weights.items():
        column = SEASON_COLUMNS[name]
        score_terms.append(f"? * pp.{column}_pctl")
        weight_params.append(weight)
        metric_columns.append(f"pp.{column}_pctl AS {column}_pctl")
        metric_columns.append(f"s.{column} AS {column}")
    query = f"""
        WITH scored AS (
            SELECT
                s.player_id,
                s.player_name AS player,
                s.teams AS team,
                s.minutes,
                100.0 * ({" + ".join(score_terms)}) AS fit_score,
                {", ".join(metric_columns)}
            FROM player_percentiles AS pp
            JOIN player_season_p90 AS s ON s.player_id = pp.player_id
            WHERE pp.position_group = ? AND s.minutes >= ?
        )
        SELECT
            ROW_NUMBER() OVER (ORDER BY fit_score DESC, minutes DESC, player ASC) AS rank,
            *
        FROM scored
        ORDER BY rank
        LIMIT ?
    """
    group, min_minutes = requirement["position_group"], requirement["min_minutes"]
    table = pd.read_sql_query(query, conn, params=(*weight_params, group, min_minutes, top_n))
    pool_size, eligible = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(s.minutes >= ?), 0)
        FROM player_percentiles AS pp
        JOIN player_season_p90 AS s ON s.player_id = pp.player_id
        WHERE pp.position_group = ?
        """,
        (min_minutes, group),
    ).fetchone()
    return FitResult(table, group, weights, min_minutes, int(pool_size), int(eligible))
