"""Scouting reports for one shortlisted player.

build_fact_sheet collects everything a report may say about a player, with
every number already formatted as a string exactly as it may appear in the
report, so the wording can later be checked against it. Functions that read
data take an open SQLite connection; nothing here prints or imports Streamlit.
"""

import sqlite3

from src.eda import metric_reliability
from src.fit import (
    ALLOWED_METRICS,
    DEFAULT_MIN_MINUTES,
    FIT_SCORE_DECIMALS,
    GROUP_LABELS,
    METRIC_NAMES,
    SEASON_COLUMNS,
    FitResult,
    fit_label,
    fit_scores,
    rank_phrase,
)

LIMITED_MINUTES = 900
SMALL_POOL = 30
LOW_RELIABILITY_SHARE = 0.5
PROFILE_METRICS = 3
PER_90_DECIMALS = 2
PERCENT_DECIMALS = 1
PERCENT_METRICS = ("pass_completion_pct",)


def _format_value(metric: str, value: float) -> tuple[str, str]:
    """Return (value, unit) as display strings for a metric's season value."""
    if metric in PERCENT_METRICS:
        return f"{value:.{PERCENT_DECIMALS}f}%", "of passes completed"
    return f"{value:.{PER_90_DECIMALS}f}", "per 90"


def _format_share(share: float) -> str:
    """Return a 0-1 share as a whole percentage, such as "31%"."""
    return f"{share * 100:.0f}%"


def _player_rows(conn: sqlite3.Connection, player_id: int) -> tuple[dict, dict]:
    """Return the player's season row and percentile row, or raise ValueError."""

    def fetch(view: str) -> dict | None:
        cursor = conn.execute(f"SELECT * FROM {view} WHERE player_id = ?", (player_id,))
        row = cursor.fetchone()
        return None if row is None else dict(zip((d[0] for d in cursor.description), row))

    season, percentiles = fetch("player_season_p90"), fetch("player_percentiles")
    if season is None or percentiles is None:
        raise ValueError(f"Player {player_id} is not in the reference pool (at least 600 minutes).")
    return season, percentiles


def _metric_entry(
    metric: str, season: dict, percentiles: dict, pool_size: int, group_label: str, tiers: dict
) -> dict:
    """Return the display fields for one metric of one player."""
    column = SEASON_COLUMNS[metric]
    percentile = percentiles[f"{column}_pctl"]
    value, unit = _format_value(metric, season[column])
    return {
        "metric": metric,
        "name": METRIC_NAMES[metric],
        "value": value,
        "unit": unit,
        "percentile": f"{round(percentile * 100)}",
        "rank": rank_phrase(percentile, pool_size, group_label),
        "reliability": tiers[metric],
    }


def _caveats(
    season: dict, fit_result: FitResult, group_label: str, tiers: dict
) -> list[str]:
    """Return the caveats that apply to this player and requirement, computed in code."""
    caveats = []
    minutes, matches = f"{season['minutes']:.0f}", f"{season['matches']}"
    if season["minutes"] < LIMITED_MINUTES:
        caveats.append(
            f"Based on limited minutes: {minutes} minutes in {matches} matches, under "
            f"{LIMITED_MINUTES} minutes, so her percentiles could move with more games."
        )
    low = [metric for metric in fit_result.weights if tiers[metric] == "low"]
    low_share = sum(fit_result.weights[metric] for metric in low)
    if low_share > LOW_RELIABILITY_SHARE:
        names = ", ".join(METRIC_NAMES[metric] for metric in low)
        caveats.append(
            f"{_format_share(low_share)} of the fit score's weight is on metrics with low "
            f"reliability at {DEFAULT_MIN_MINUTES} minutes ({names}), so the score is a weaker "
            "signal than it looks."
        )
    if fit_result.pool_size < SMALL_POOL:
        caveats.append(
            f"Small reference pool: her percentiles compare her with only {fit_result.pool_size} "
            f"{group_label}, so one place in a ranking is a big step in percentile."
        )
    teams = season["teams"].split(",")
    if len(teams) > 1:
        caveats.append(
            f"Her season figures combine her matches for {' and '.join(teams)} into one row, "
            "after a mid-season move."
        )
    return caveats


def build_fact_sheet(conn: sqlite3.Connection, fit_result: FitResult, player_id: int) -> dict:
    """Return every fact a scouting report may use about one shortlisted player.

    The player must be eligible under fit_result (in its position group with
    at least its min_minutes). Every number is a preformatted string: minutes
    and matches as whole numbers, the fit score to one decimal, per-90 values
    to two decimals, pass completion as a percentage to one decimal,
    percentiles as whole numbers from 0 to 100 and weights as each metric's
    whole-percent share of the fit score. Reliability tiers are those at 600
    minutes. Strengths and weaknesses are the three highest and three lowest
    percentiles across all allowed metrics, ties broken in metric order.
    """
    season, percentiles = _player_rows(conn, player_id)
    group = fit_result.position_group
    if season["position_group"] != group or season["minutes"] < fit_result.min_minutes:
        raise ValueError(
            f"{season['player_name']} is not eligible for this shortlist "
            f"({group}, at least {fit_result.min_minutes} minutes)."
        )
    requirement = {
        "position_group": group,
        "weights": fit_result.weights,
        "min_minutes": fit_result.min_minutes,
    }
    ranking = fit_scores(conn, requirement, top_n=fit_result.eligible_players).table
    row = ranking.loc[ranking["player_id"] == player_id].iloc[0]

    group_label = GROUP_LABELS[group]
    reliability = metric_reliability(conn, DEFAULT_MIN_MINUTES)
    tiers = dict(zip(reliability["metric"], reliability["tier"]))
    pool = fit_result.pool_size

    def entry(metric: str) -> dict:
        return _metric_entry(metric, season, percentiles, pool, group_label, tiers)

    weighted = [
        {**entry(metric), "weight": _format_share(weight)} for metric, weight in fit_result.weights.items()
    ]
    order = {metric: index for index, metric in enumerate(ALLOWED_METRICS)}
    by_percentile = sorted(
        ALLOWED_METRICS,
        key=lambda metric: (-percentiles[f"{SEASON_COLUMNS[metric]}_pctl"], order[metric]),
    )
    lowest_first = sorted(
        ALLOWED_METRICS,
        key=lambda metric: (percentiles[f"{SEASON_COLUMNS[metric]}_pctl"], order[metric]),
    )
    score = row["fit_score"]
    return {
        "player": {
            "name": season["player_name"],
            "teams": season["teams"].split(","),
            "position_group": group,
            "minutes": f"{season['minutes']:.0f}",
            "matches": f"{season['matches']}",
        },
        "shortlist": {
            "rank": f"{int(row['rank'])}",
            "eligible_players": f"{fit_result.eligible_players}",
            "reference_pool": f"{pool}",
            "min_minutes": f"{fit_result.min_minutes}",
        },
        "fit": {"score": f"{score:.{FIT_SCORE_DECIMALS}f}", "label": fit_label(score)},
        "weighted_metrics": weighted,
        "strengths": [entry(metric) for metric in by_percentile[:PROFILE_METRICS]],
        "weaknesses": [entry(metric) for metric in lowest_first[:PROFILE_METRICS]],
        "caveats": _caveats(season, fit_result, group_label, tiers),
    }
