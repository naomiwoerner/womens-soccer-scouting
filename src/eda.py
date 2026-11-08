"""Exploratory analysis of the scouting database.

Every function takes an open SQLite connection and returns a DataFrame or a
Plotly figure. Nothing here reads data/raw, so the app can run from the
database alone.
"""

import re
import sqlite3
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.metrics import METRICS

DEFAULT_THRESHOLDS = (300, 400, 600, 900, 1200)
STABILITY_THRESHOLDS = (200, 300, 400, 600, 900, 1200)
STABILITY_DRAWS = 200
STABILITY_SEED = 42
POOL_HALF_MINUTES = 600
RATE_DECIMALS = 9
DEFAULT_MIN_MINUTES = 600
OUTLIER_IQR_MULTIPLE = 1.5
OUTLIER_MINUTES_BAND = 900
SEASON_MATCHES = 22
PLAUSIBLE_ORDER_R = 0.80
TOTALS_TOLERANCE = 1e-6
SCOPE_COLUMNS = {"pooled": "pooled_r", "within_group": "within_group_r"}
CORRELATION_METRICS = [metric for metric in METRICS if metric.name != "xg"]
HIGH_RELIABILITY = 0.70
MODERATE_RELIABILITY = 0.55
RELIABILITY_NOTES = {
    "high": (
        "Stable: players tend to keep their rank on this metric from one sample of matches "
        "to another, so differences between players are meaningful."
    ),
    "moderate": (
        "Fairly stable: useful evidence, but small differences between players may be noise, "
        "so read it alongside other metrics."
    ),
    "low": (
        "Unstable at this sample size: a player's rank can change a lot depending on which "
        "matches are sampled, so treat it as a weak signal and do not rely on it alone."
    ),
}
MIN_GROUP_PLAYERS = 8
FIRST = "first"
SECOND = "second"
HISTOGRAM_BIN_MINUTES = 100

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
CONTEXT_LINE = "#c3c2b7"
BAND_FILL = "rgba(42, 120, 214, 0.22)"
DIVERGING_NEGATIVE = "#e34948"
DIVERGING_MIDPOINT = "#f0efec"
DIVERGING_POSITIVE = "#2a78d6"
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4")


def _position_groups(conn: sqlite3.Connection) -> list[str]:
    """Return the position groups in a fixed order, so each keeps its chart color."""
    query = """
        SELECT DISTINCT position_group
        FROM players
        WHERE position_group IS NOT NULL
        ORDER BY position_group
    """
    return pd.read_sql_query(query, conn)["position_group"].tolist()


def _outfield_season_minutes(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return each outfield player's total season minutes and position group."""
    query = """
        SELECT pm.player_id, p.position_group, SUM(pm.minutes) AS minutes
        FROM player_match AS pm
        JOIN players AS p ON p.player_id = pm.player_id
        WHERE p.position_group IS NOT NULL
        GROUP BY pm.player_id, p.position_group
    """
    return pd.read_sql_query(query, conn)


def _season_view_minimum(conn: sqlite3.Connection) -> int:
    """Return the minutes minimum written into the player_season_p90 view."""
    query = "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'player_season_p90'"
    view_sql = conn.execute(query).fetchone()[0]
    return int(re.search(r"HAVING SUM\(pm\.minutes\) >= (\d+)", view_sql).group(1))


def _completion_check_rows(conn: sqlite3.Connection) -> list[dict]:
    """Return the checks on how pass completion % is calculated.

    The zero-pass rows are harmless only if season completion is built from
    totals. This recomputes each season-view player both from totals and as an
    average of match percentages, then compares the view with each, so the
    result shows the view's actual behaviour and that the two methods differ.
    """
    zero_pass_rows = conn.execute(
        "SELECT COUNT(*) FROM player_match WHERE pass_completion_pct IS NULL"
    ).fetchone()[0]
    query = """
        SELECT
            v.pass_completion_pct AS view_pct,
            SUM(pm.passes_completed) * 100.0 / SUM(pm.passes_attempted) AS totals_pct,
            AVG(pm.pass_completion_pct) AS match_average_pct
        FROM player_season_p90 AS v
        JOIN player_match AS pm ON pm.player_id = v.player_id
        GROUP BY v.player_id
    """
    season = pd.read_sql_query(query, conn)
    differs_from_totals = int(((season["view_pct"] - season["totals_pct"]).abs() > 1e-9).sum())
    differs_from_average = int(
        ((season["view_pct"] - season["match_average_pct"]).abs() > 1e-9).sum()
    )
    return [
        {
            "check": "Player-match rows with zero passes (null completion %)",
            "count": zero_pass_rows,
            "treatment": (
                "Stored as NULL, never 0. Season completion % is computed from total completed / "
                "total attempted, never as an average of match percentages."
            ),
        },
        {
            "check": "Season-view players whose completion % differs from completed / attempted",
            "count": differs_from_totals,
            "treatment": (
                f"Verified against the view's values across {len(season)} players: zero means it "
                f"uses totals. {differs_from_average} of them differ from an average of match "
                "percentages, so this check can tell the two methods apart. player_percentiles "
                "ranks this same column."
            ),
        },
    ]


def _transfer_check_row(conn: sqlite3.Connection) -> dict:
    """Return the check on players who appeared for more than one club."""
    query = """
        SELECT p.player_name
        FROM player_match AS pm
        JOIN players AS p ON p.player_id = pm.player_id
        GROUP BY pm.player_id
        HAVING COUNT(DISTINCT pm.team_id) > 1
        ORDER BY p.player_name
    """
    names = pd.read_sql_query(query, conn)["player_name"].tolist()
    return {
        "check": "Players with more than one club (transfers)",
        "count": len(names),
        "treatment": (
            "Combined into one season row: minutes and counts are summed across clubs, the view "
            "lists both clubs, and the position group uses minutes at both. "
            f"Players: {', '.join(names)}."
        ),
    }


def _exclusion_check_rows(conn: sqlite3.Connection) -> list[dict]:
    """Return the counts of players left out of the season views, and why."""
    goalkeepers, goalkeeper_rows = conn.execute(
        """
        SELECT COUNT(DISTINCT pm.player_id), COUNT(*)
        FROM player_match AS pm
        JOIN players AS p ON p.player_id = pm.player_id
        WHERE p.position_group IS NULL
        """
    ).fetchone()
    minimum = _season_view_minimum(conn)
    minutes = _outfield_season_minutes(conn)
    return [
        {
            "check": "Goalkeepers excluded",
            "count": goalkeepers,
            "treatment": (
                f"No position group, so not in the season views. Their {goalkeeper_rows} "
                "player-match rows stay in player_match so totals still match the raw events."
            ),
        },
        {
            "check": f"Outfield players below {minimum} minutes excluded",
            "count": int((minutes["minutes"] < minimum).sum()),
            "treatment": (
                "Left out of player_season_p90 and player_percentiles, but kept in player_match. "
                f"{len(minutes)} outfield players appeared at all."
            ),
        },
    ]


def data_quality_summary(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return every data-quality check as columns check, count and treatment.

    A reader needs to know what is missing or excluded, and what was done about
    it, before trusting any ranking. Checks that need the raw events come from
    the data_quality table written by scripts/build_db.py. The rest are computed
    here from the database, so they always describe the current data.
    """
    stored = pd.read_sql_query(
        'SELECT check_name AS "check", count, treatment FROM data_quality ORDER BY check_name',
        conn,
    )
    missing_position = conn.execute(
        """
        SELECT COUNT(*)
        FROM player_match AS pm
        LEFT JOIN players AS p ON p.player_id = pm.player_id
        WHERE p.primary_position IS NULL
        """
    ).fetchone()[0]
    computed = [
        {
            "check": "Player-match rows with missing position",
            "count": missing_position,
            "treatment": (
                "Every appearance has a position from the lineup stints. Goalkeepers have a "
                "position but no group, and are counted below."
            ),
        },
        *_completion_check_rows(conn),
        _transfer_check_row(conn),
        *_exclusion_check_rows(conn),
        _outlier_check_row(conn),
    ]
    return pd.concat([stored, pd.DataFrame(computed)], ignore_index=True)


def minutes_threshold_table(
    conn: sqlite3.Connection, thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Return how many outfield players reach each season-minutes threshold.

    Each row has the threshold, the total, and one count per position group.
    This shows the trade-off behind a minimum: a higher one gives steadier
    per-90 numbers but leaves fewer players to compare within each group.
    """
    groups = _position_groups(conn)
    minutes = _outfield_season_minutes(conn)
    rows = []
    for threshold in thresholds:
        qualifying = minutes[minutes["minutes"] >= threshold]
        by_group = qualifying.groupby("position_group").size().reindex(groups, fill_value=0)
        rows.append({"threshold": threshold, "total": len(qualifying), **by_group.to_dict()})
    return pd.DataFrame(rows)


def _last_week_of_first_half(conn: sqlite3.Connection) -> int:
    """Return the final match week of the season's first half."""
    weeks = pd.read_sql_query(
        "SELECT DISTINCT match_week FROM matches ORDER BY match_week", conn
    )["match_week"].tolist()
    return int(weeks[len(weeks) // 2 - 1])


def season_split(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the weeks, match count and dates in each half of the season.

    The stability analysis compares a player's first and second halves, so the
    split is reported to show that it is even and where it falls in the calendar.
    Match week is used because every match has one.
    """
    query = """
        SELECT
            CASE WHEN match_week <= ? THEN 'First half' ELSE 'Second half' END AS half,
            MIN(match_week) AS first_week,
            MAX(match_week) AS last_week,
            COUNT(*) AS matches,
            MIN(match_date) AS first_date,
            MAX(match_date) AS last_date
        FROM matches
        GROUP BY 1
        ORDER BY 1
    """
    return pd.read_sql_query(query, conn, params=(_last_week_of_first_half(conn),))


@dataclass(frozen=True)
class _StabilityPool:
    """The fixed players in the stability analysis and their match-by-match numbers.

    Each entry of first and second is one player's matches in that half, as an
    array with one row per match and one column per name in column_index.
    """

    groups: np.ndarray
    first: list[np.ndarray]
    second: list[np.ndarray]
    column_index: dict[str, int]


def _pool_match_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return match rows for outfield players with enough minutes in each half.

    The pool needs POOL_HALF_MINUTES in both halves, the amount required to
    evaluate the highest threshold. Rows are sorted so that a seed always
    produces the same draws.
    """
    sums = ", ".join(f"pm.{column}" for column in _stability_columns())
    query = f"""
        SELECT
            pm.player_id,
            p.position_group,
            CASE WHEN m.match_week <= ? THEN '{FIRST}' ELSE '{SECOND}' END AS half,
            {sums}
        FROM player_match AS pm
        JOIN players AS p ON p.player_id = pm.player_id
        JOIN matches AS m ON m.match_id = pm.match_id
        WHERE p.position_group IS NOT NULL
        ORDER BY pm.player_id, pm.match_id
    """
    rows = pd.read_sql_query(query, conn, params=(_last_week_of_first_half(conn),))
    half_minutes = rows.groupby(["player_id", "half"])["minutes"].sum().unstack()
    in_pool = half_minutes[
        (half_minutes[FIRST] >= POOL_HALF_MINUTES) & (half_minutes[SECOND] >= POOL_HALF_MINUTES)
    ].index
    return rows[rows["player_id"].isin(in_pool)]


def _stability_columns() -> list[str]:
    """Return the player_match columns the stability samples add up."""
    return ["minutes", "passes_attempted", *[m.name for m in METRICS if m.per_90]]


def stability_pool_sizes(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the fixed stability pool's size per position group, with a total row.

    Every threshold uses these same players, so differences between thresholds
    come from minutes rather than from who is in the sample. A group below
    MIN_GROUP_PLAYERS is marked as not included.
    """
    sizes = _pool_match_rows(conn).groupby("position_group")["player_id"].nunique()
    table = sizes.rename("n_players").reset_index()
    table["included"] = table["n_players"] >= MIN_GROUP_PLAYERS
    total = pd.DataFrame(
        [{"position_group": "Total", "n_players": int(sizes.sum()), "included": True}]
    )
    return pd.concat([table, total], ignore_index=True)


def stability_skipped_groups(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the position groups left out of the stability pool for having n < 8.

    A rank correlation on so few players is mostly noise. Recording what was
    dropped shows which position groups the stability averages do not cover.
    """
    sizes = stability_pool_sizes(conn)
    return sizes.loc[~sizes["included"], ["position_group", "n_players"]].reset_index(drop=True)


def _stability_pool(conn: sqlite3.Connection) -> _StabilityPool:
    """Return the pool as arrays, leaving out groups that are too small."""
    columns = _stability_columns()
    rows = _pool_match_rows(conn)
    group_sizes = rows.groupby("position_group")["player_id"].transform("nunique")
    rows = rows[group_sizes >= MIN_GROUP_PLAYERS]

    groups, first, second = [], [], []
    for _, player in rows.groupby("player_id", sort=True):
        groups.append(player["position_group"].iloc[0])
        first.append(player.loc[player["half"] == FIRST, columns].to_numpy(dtype=float))
        second.append(player.loc[player["half"] == SECOND, columns].to_numpy(dtype=float))
    column_index = {name: position for position, name in enumerate(columns)}
    return _StabilityPool(np.array(groups), first, second, column_index)


def _sample_rates(totals: np.ndarray, column_index: dict[str, int]) -> np.ndarray:
    """Return per-90 values for sampled totals, one row per player, one column per metric.

    Completion % is built from the sample's totals rather than match percentages.
    """
    minutes = totals[:, column_index["minutes"]]
    rates = np.empty((len(totals), len(METRICS)))
    for position, metric in enumerate(METRICS):
        if metric.per_90:
            rates[:, position] = totals[:, column_index[metric.name]] * 90 / minutes
        else:
            completed = totals[:, column_index["passes_completed"]]
            rates[:, position] = 100 * completed / totals[:, column_index["passes_attempted"]]
    # Adding minutes in a different order can change the last floating-point digits,
    # which would split true ties when ranking.
    return rates.round(RATE_DECIMALS)


def _column_spearman(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the Spearman correlation between matching columns of two arrays.

    Spearman is the Pearson correlation of average ranks, which handles ties.
    """
    first_ranks = pd.DataFrame(first).rank().to_numpy()
    second_ranks = pd.DataFrame(second).rank().to_numpy()
    first_ranks = first_ranks - first_ranks.mean(axis=0)
    second_ranks = second_ranks - second_ranks.mean(axis=0)
    covariance = (first_ranks * second_ranks).sum(axis=0)
    return covariance / np.sqrt((first_ranks**2).sum(axis=0) * (second_ranks**2).sum(axis=0))


def _weighted_split_correlations(
    first: np.ndarray, second: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Return each metric's n-weighted average of within-group Spearman correlations.

    Correlating within groups stops the gap between positions (strikers shoot
    more than center backs in both halves) from inflating the result.
    """
    weighted = np.zeros(first.shape[1])
    for group in np.unique(groups):
        members = groups == group
        weighted += _column_spearman(first[members], second[members]) * members.sum()
    return weighted / len(groups)


def _spearman_brown(r_split: np.ndarray) -> np.ndarray:
    """Return the Spearman-Brown estimate 2r / (1 + r), undefined at r = -1."""
    return np.where(r_split > -1, 2 * r_split / (1 + r_split), np.nan)


def _stability_draws(
    conn: sqlite3.Connection, thresholds: tuple[int, ...], n_draws: int, seed: int
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return per-draw correlations and the actual minutes used in every sample.

    In each draw, every player's matches in each half are shuffled once and
    added up until the minutes reach T/2. Sharing that order across thresholds
    means a higher threshold extends the same sample, so differences between
    thresholds are not blurred by unrelated random draws.

    The DataFrame has draw, threshold, metric, n_players, r_split and r_full. The
    array of actual minutes is indexed by draw, threshold, player and half.
    """
    pool = _stability_pool(conn)
    targets = np.array(thresholds, dtype=float) / 2
    rng = np.random.default_rng(seed)
    n_players = len(pool.groups)
    n_columns = len(pool.column_index)
    minutes_column = pool.column_index["minutes"]
    actual_minutes = np.empty((n_draws, len(thresholds), n_players, 2))

    records = []
    for draw in range(n_draws):
        totals = np.empty((2, len(thresholds), n_players, n_columns))
        for half, players in enumerate((pool.first, pool.second)):
            for player, matches in enumerate(players):
                running = matches[rng.permutation(len(matches))].cumsum(axis=0)
                reached = np.searchsorted(running[:, minutes_column], targets)
                totals[half, :, player] = running[np.minimum(reached, len(running) - 1)]
        actual_minutes[draw] = np.moveaxis(totals[:, :, :, minutes_column], 0, -1)

        for position, threshold in enumerate(thresholds):
            first = _sample_rates(totals[0, position], pool.column_index)
            second = _sample_rates(totals[1, position], pool.column_index)
            r_split = _weighted_split_correlations(first, second, pool.groups)
            r_full = _spearman_brown(r_split)
            records.extend(
                (draw, threshold, metric.name, n_players, r_split[index], r_full[index])
                for index, metric in enumerate(METRICS)
            )
    columns = ["draw", "threshold", "metric", "n_players", "r_split", "r_full"]
    return pd.DataFrame(records, columns=columns), actual_minutes


def _metric_summary(draws: pd.DataFrame) -> pd.DataFrame:
    """Return each metric's average r_split and r_full and r_full's 10th and 90th percentiles."""
    grouped = draws.groupby(["threshold", "metric"], sort=False)
    summary = grouped.agg(
        n_players=("n_players", "first"),
        r_split=("r_split", "mean"),
        r_full=("r_full", "mean"),
        r_full_p10=("r_full", lambda values: values.quantile(0.1)),
        r_full_p90=("r_full", lambda values: values.quantile(0.9)),
    )
    return summary.reset_index()


def _mean_summary(draws: pd.DataFrame) -> pd.DataFrame:
    """Return the all-metrics mean r_full per threshold and its range across draws.

    The range comes from each draw's mean, because averaging the per-metric
    percentiles would overstate it: metrics do not all move together.
    """
    per_draw = draws.groupby(["draw", "threshold"], as_index=False)["r_full"].mean()
    return (
        per_draw.groupby("threshold")["r_full"]
        .agg(
            mean_r_full="mean",
            p10=lambda values: values.quantile(0.1),
            p90=lambda values: values.quantile(0.9),
        )
        .reset_index()
    )


def stability_by_threshold(
    conn: sqlite3.Connection,
    thresholds: tuple[int, ...] = STABILITY_THRESHOLDS,
    n_draws: int = STABILITY_DRAWS,
    seed: int = STABILITY_SEED,
) -> pd.DataFrame:
    """Return how reliable every metric is at about T minutes of data.

    A metric only helps a scout if it measures something repeatable about the
    player. The cap is needed because a minutes threshold alone is only a floor:
    players above it usually have far more minutes, so an uncapped correlation
    describes their whole sample rather than T minutes. Here every player in one
    fixed pool contributes about T/2 minutes per half, in whole matches drawn at
    random, and the within-group Spearman correlation is repeated over many draws.

    Returns threshold, metric, n_players, r_split (reliability at about T/2),
    r_full (Spearman-Brown estimate at about T, averaged over draws) and
    r_full_p10 and r_full_p90 (its 10th and 90th percentiles across draws).
    """
    draws, _ = _stability_draws(conn, thresholds, n_draws, seed)
    return _metric_summary(draws)


def stability_mean_range(
    conn: sqlite3.Connection,
    thresholds: tuple[int, ...] = STABILITY_THRESHOLDS,
    n_draws: int = STABILITY_DRAWS,
    seed: int = STABILITY_SEED,
) -> pd.DataFrame:
    """Return the mean r_full across metrics per threshold, with its 10-90% range.

    This single summary is what a minimum-minutes decision rests on. The range
    shows how much it moves with the random choice of matches.
    """
    draws, _ = _stability_draws(conn, thresholds, n_draws, seed)
    return _mean_summary(draws)


def median_minutes_per_half(
    conn: sqlite3.Connection,
    thresholds: tuple[int, ...] = STABILITY_THRESHOLDS,
    n_draws: int = STABILITY_DRAWS,
    seed: int = STABILITY_SEED,
) -> pd.DataFrame:
    """Return the median minutes actually used per half at each threshold.

    Whole matches overshoot the T/2 target, so this shows how much data each
    capped sample really contains, pooled over all players and draws.
    """
    _, actual = _stability_draws(conn, thresholds, n_draws, seed)
    return _median_minutes_table(actual, thresholds)


def _median_minutes_table(actual: np.ndarray, thresholds: tuple[int, ...]) -> pd.DataFrame:
    """Return median actual minutes per half from the array of sampled minutes."""
    rows = []
    for position, threshold in enumerate(thresholds):
        minutes = actual[:, position]
        rows.append(
            {
                "threshold": threshold,
                "half_target": threshold / 2,
                "n_players": minutes.shape[1],
                "median_first_half": float(np.median(minutes[:, :, 0])),
                "median_second_half": float(np.median(minutes[:, :, 1])),
                "median_per_half": float(np.median(minutes)),
            }
        )
    return pd.DataFrame(rows)


def _reliability_tier(r_full: float) -> str:
    """Return the reliability tier for an r_full value."""
    if r_full >= HIGH_RELIABILITY:
        return "high"
    if r_full >= MODERATE_RELIABILITY:
        return "moderate"
    return "low"


def metric_reliability(conn: sqlite3.Connection, threshold: int = 600) -> pd.DataFrame:
    """Return each metric's reliability at a minutes threshold, with a tier and note.

    The minimum-minutes rule is a trade-off, so readers need to know which
    metrics can be trusted at that sample size. Returns metric, r_full, p10, p90,
    tier and note. Tiers come from the computed r_full, so they follow the data.
    The threshold cannot exceed what the fixed pool guarantees in each half.
    """
    if not 0 < threshold <= 2 * POOL_HALF_MINUTES:
        raise ValueError(f"threshold must be above 0 and at most {2 * POOL_HALF_MINUTES}")
    stability = stability_by_threshold(conn, thresholds=(threshold,))
    reliability = stability.rename(columns={"r_full_p10": "p10", "r_full_p90": "p90"})[
        ["metric", "r_full", "p10", "p90"]
    ]
    reliability["tier"] = reliability["r_full"].map(_reliability_tier)
    reliability["note"] = reliability["tier"].map(RELIABILITY_NOTES)
    return reliability


def reliability_at_minutes(conn: sqlite3.Connection, minutes: float) -> pd.DataFrame:
    """Return the mean r_full across metrics for a given number of minutes.

    The app's minutes slider needs a value between the computed thresholds. This
    interpolates linearly between them and holds the end values outside their
    range. The x-values are actual sampled minutes (2 x the median actual minutes
    per half), not the nominal thresholds, because whole matches overshoot each
    target and nominal labels would make the slider value optimistic. Returns one
    row with minutes and mean_r_full.
    """
    thresholds = STABILITY_THRESHOLDS
    draws, actual = _stability_draws(conn, thresholds, STABILITY_DRAWS, STABILITY_SEED)
    sampled_minutes = 2 * _median_minutes_table(actual, thresholds)["median_per_half"]
    means = _mean_summary(draws)["mean_r_full"]
    mean_r_full = float(np.interp(minutes, sampled_minutes, means))
    return pd.DataFrame([{"minutes": minutes, "mean_r_full": mean_r_full}])


def _chart_layout(title: str, x_title: str, y_title: str, show_legend: bool) -> dict:
    """Return the layout shared by every figure: quiet axes on a light surface."""
    axis = {
        "showline": False,
        "ticks": "",
        "tickfont": {"color": TEXT_SECONDARY},
        "title_font": {"color": TEXT_SECONDARY},
        "zeroline": False,
    }
    return {
        "title": {
            "text": title,
            "x": 0,
            "xref": "paper",
            "xanchor": "left",
            "y": 0.97,
            "yref": "container",
            "yanchor": "top",
            "font": {"color": TEXT_PRIMARY, "size": 18},
        },
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": "system-ui, -apple-system, Segoe UI, sans-serif", "color": TEXT_PRIMARY},
        "xaxis": {**axis, "title": {"text": x_title}, "showgrid": False},
        "yaxis": {**axis, "title": {"text": y_title}, "showgrid": True, "gridcolor": GRID},
        "showlegend": show_legend,
        "legend": {
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
            "font": {"color": TEXT_SECONDARY},
        },
        "hoverlabel": {"bgcolor": SURFACE, "font": {"color": TEXT_PRIMARY}},
        "margin": {"l": 64, "r": 72, "t": 96, "b": 64},
    }


def minutes_histogram_figure(
    conn: sqlite3.Connection, thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
) -> go.Figure:
    """Return a histogram of outfield season minutes with a line at each threshold.

    It shows where each candidate minimum cuts the player pool. Bins are 100
    minutes wide, so every threshold falls on a bin edge.
    """
    minutes = _outfield_season_minutes(conn)["minutes"]
    figure = go.Figure(
        go.Histogram(
            x=minutes,
            xbins={"start": 0, "end": float(minutes.max()) + HISTOGRAM_BIN_MINUTES, "size": HISTOGRAM_BIN_MINUTES},
            marker={"color": SERIES_COLORS[0], "cornerradius": 4, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{x} minutes<br>%{y} players<extra></extra>",
        )
    )
    for threshold in thresholds:
        figure.add_vline(
            x=threshold,
            line={"color": TEXT_SECONDARY, "width": 1, "dash": "dash"},
            annotation={"text": str(threshold), "font": {"color": TEXT_SECONDARY, "size": 12}},
            annotation_position="top",
        )
    figure.update_layout(
        **_chart_layout("Season minutes, outfield players", "Season minutes", "Players", False)
    )
    return figure


def reliability_figure(
    conn: sqlite3.Connection,
    thresholds: tuple[int, ...] = STABILITY_THRESHOLDS,
    n_draws: int = STABILITY_DRAWS,
    seed: int = STABILITY_SEED,
) -> go.Figure:
    """Return capped r_full against threshold, with faint metrics, a mean and its band.

    It shows whether more minutes buy meaningfully better reliability. The metric
    lines share one quiet color and name themselves on hover, because 16 separate
    hues could not be told apart. The band is the 10-90% range of the mean across
    random draws of matches, so it does not cover uncertainty from having a
    limited number of players.
    """
    draws, _ = _stability_draws(conn, thresholds, n_draws, seed)
    stability = _metric_summary(draws)
    mean = _mean_summary(draws)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=mean["threshold"],
            y=mean["p90"],
            mode="lines",
            line={"width": 0},
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=mean["threshold"],
            y=mean["p10"],
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor=BAND_FILL,
            name="10-90% of draws (mean)",
            hoverinfo="skip",
        )
    )
    for index, (metric, rows) in enumerate(stability.groupby("metric", sort=False)):
        figure.add_trace(
            go.Scatter(
                x=rows["threshold"],
                y=rows["r_full"],
                mode="lines",
                name="Individual metric",
                legendgroup="metrics",
                showlegend=index == 0,
                line={"color": CONTEXT_LINE, "width": 1},
                hovertemplate=f"{metric}<br>T = %{{x}}<br>r_full = %{{y:.2f}}<extra></extra>",
            )
        )
    figure.add_trace(
        go.Scatter(
            x=mean["threshold"],
            y=mean["mean_r_full"],
            mode="lines+markers",
            name="Mean of all metrics",
            line={"color": SERIES_COLORS[0], "width": 3},
            marker={"color": SERIES_COLORS[0], "size": 9, "line": {"color": SURFACE, "width": 2}},
            customdata=mean[["p10", "p90"]],
            hovertemplate=(
                "Mean<br>T = %{x}<br>r_full = %{y:.2f}"
                "<br>10-90%: %{customdata[0]:.2f} to %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    figure.add_annotation(
        x=mean["threshold"].iloc[-1],
        y=mean["mean_r_full"].iloc[-1],
        text="Mean",
        showarrow=False,
        xanchor="left",
        xshift=10,
        font={"color": TEXT_PRIMARY, "size": 12},
    )
    figure.update_layout(
        **_chart_layout(
            "Reliability at about T minutes of data",
            "Minutes of data (T)",
            "Reliability (r_full)",
            True,
        )
    )
    figure.update_xaxes(tickvals=list(thresholds))
    figure.update_yaxes(range=[min(0.0, float(stability["r_full"].min())), 1.0])
    return figure


def qualifying_players_figure(
    conn: sqlite3.Connection, thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
) -> go.Figure:
    """Return grouped bars of qualifying players per position group and threshold.

    It shows which positions run short of comparison players first as the
    minimum rises. Each group keeps the same color at every threshold.
    """
    table = minutes_threshold_table(conn, thresholds)
    figure = go.Figure()
    for group, color in zip(_position_groups(conn), SERIES_COLORS):
        figure.add_trace(
            go.Bar(
                x=table["threshold"].astype(str),
                y=table[group],
                name=group,
                marker={"color": color, "cornerradius": 4, "line": {"color": SURFACE, "width": 2}},
                hovertemplate=f"{group}<br>T = %{{x}}<br>%{{y}} players<extra></extra>",
            )
        )
    figure.update_layout(
        **_chart_layout(
            "Qualifying players by position group", "Minutes threshold (T)", "Players", True
        ),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.05,
    )
    # Five group names wrap the legend onto two rows, so leave room under the title.
    figure.update_layout(margin={"t": 136})
    return figure


def _season_column(metric: str) -> str:
    """Return a metric's column in the season table, rejecting unknown names."""
    for candidate in METRICS:
        if candidate.name == metric:
            return candidate.season_column
    raise ValueError(f"Unknown metric: {metric}")


def _season_rates(
    conn: sqlite3.Connection, min_minutes: float = DEFAULT_MIN_MINUTES, group: str | None = None
) -> pd.DataFrame:
    """Return per-player season rates for any minutes minimum, optionally for one group.

    This matches player_season_p90, whose minimum is fixed at 600, so that the
    app's slider can reuse the same analysis at other minimums. Completion % comes
    from season totals.
    """
    if group is not None and group not in _position_groups(conn):
        raise ValueError(f"Unknown position group: {group}")
    rates = []
    for metric in METRICS:
        if metric.per_90:
            rates.append(f"SUM(pm.{metric.name}) * 90.0 / SUM(pm.minutes) AS {metric.season_column}")
        else:
            rates.append(
                "SUM(pm.passes_completed) * 100.0 / NULLIF(SUM(pm.passes_attempted), 0)"
                f" AS {metric.season_column}"
            )
    query = f"""
        SELECT
            p.player_id,
            p.player_name,
            p.position_group,
            GROUP_CONCAT(DISTINCT t.team_name) AS teams,
            COUNT(*) AS matches,
            SUM(pm.minutes) AS minutes,
            {", ".join(rates)}
        FROM player_match AS pm
        JOIN players AS p ON p.player_id = pm.player_id
        JOIN teams AS t ON t.team_id = pm.team_id
        WHERE p.position_group IS NOT NULL
        GROUP BY p.player_id, p.player_name, p.position_group
        HAVING SUM(pm.minutes) >= ?
    """
    season = pd.read_sql_query(query, conn, params=(min_minutes,))
    return season if group is None else season[season["position_group"] == group]


def descriptive_stats(
    conn: sqlite3.Connection, group: str | None = None, min_minutes: float = DEFAULT_MIN_MINUTES
) -> pd.DataFrame:
    """Return n, mean, median, std, min, p10, p90 and max per position group and metric.

    Each group is summarised on its own because positions differ so much that a
    league-wide average would describe nobody. Pass group to get just one.
    """
    rows = []
    for position_group, players in _season_rates(conn, min_minutes, group).groupby("position_group"):
        for metric in METRICS:
            values = players[metric.season_column]
            rows.append(
                {
                    "position_group": position_group,
                    "metric": metric.name,
                    "n": int(values.count()),
                    "mean": values.mean(),
                    "median": values.median(),
                    "std": values.std(),
                    "min": values.min(),
                    "p10": values.quantile(0.1),
                    "p90": values.quantile(0.9),
                    "max": values.max(),
                }
            )
    return pd.DataFrame(rows)


def metric_distribution_fig(
    conn: sqlite3.Connection, metric: str, min_minutes: float = DEFAULT_MIN_MINUTES
) -> go.Figure:
    """Return a box plot of one metric by position group, with every player shown.

    Showing the points reveals how many players sit behind each box and who the
    extremes are. Hovering a point gives the player, team and minutes. Boxes share
    one color because the axis already names each group.
    """
    column = _season_column(metric)
    rates = _season_rates(conn, min_minutes)
    figure = go.Figure()
    for group in _position_groups(conn):
        players = rates[rates["position_group"] == group]
        figure.add_trace(
            go.Box(
                x=players[column],
                name=group,
                orientation="h",
                boxpoints="all",
                jitter=0.45,
                pointpos=0,
                hoveron="points",
                customdata=players[["player_name", "teams", "minutes"]].to_numpy(),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]:.0f} minutes"
                    f"<br>{metric}: %{{x:.2f}}<extra></extra>"
                ),
                marker={
                    "color": SERIES_COLORS[0],
                    "size": 8,
                    "opacity": 0.6,
                    "line": {"color": SURFACE, "width": 1},
                },
                line={"color": SERIES_COLORS[0], "width": 2},
                fillcolor=BAND_FILL,
            )
        )
    figure.update_layout(**_chart_layout(f"{column} by position group", column, "", False))
    figure.update_xaxes(showgrid=True, gridcolor=GRID)
    figure.update_yaxes(showgrid=False, autorange="reversed", automargin=True)
    return figure


def outlier_table(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return players outside the Tukey fences, and a summary of those flags.

    Within each group and metric, a player is flagged outside Q1 - 1.5 * IQR or
    Q3 + 1.5 * IQR. Treatment: outliers are kept, because they are real
    performances and percentiles are rank-based and robust to them, but any
    concentration among low-minute players is noted.

    The first DataFrame has player, team, group, metric, value, minutes and
    direction. The second has, per metric and overall, the number of flags, the
    share from players with 600-900 minutes and over 900 minutes, and the share of
    the 600+ pool in the 600-900 band, which is the baseline for judging concentration.
    """
    season = pd.read_sql_query("SELECT * FROM player_season_p90", conn)
    flags = []
    for group, players in season.groupby("position_group"):
        for metric in METRICS:
            values = players[metric.season_column]
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            spread = OUTLIER_IQR_MULTIPLE * (q3 - q1)
            for direction, outside in (("high", values > q3 + spread), ("low", values < q1 - spread)):
                for _, player in players[outside].iterrows():
                    flags.append(
                        {
                            "player": player["player_name"],
                            "team": player["teams"],
                            "group": group,
                            "metric": metric.name,
                            "value": player[metric.season_column],
                            "minutes": player["minutes"],
                            "direction": direction,
                        }
                    )
    table = pd.DataFrame(flags)
    pool_share = float((season["minutes"] <= OUTLIER_MINUTES_BAND).mean())

    def summary_row(label: str, rows: pd.DataFrame) -> dict:
        in_band = float((rows["minutes"] <= OUTLIER_MINUTES_BAND).mean()) if len(rows) else np.nan
        return {
            "metric": label,
            "n_flags": len(rows),
            "share_600_900": in_band,
            "share_over_900": 1 - in_band,
            "pool_share_600_900": pool_share,
        }

    summary = [summary_row(m.name, table[table["metric"] == m.name]) for m in METRICS]
    summary.append(summary_row("ALL", table))
    return table, pd.DataFrame(summary)


def _outlier_check_row(conn: sqlite3.Connection) -> dict:
    """Return the data-quality row documenting how outliers are treated."""
    _, summary = outlier_table(conn)
    overall = summary[summary["metric"] == "ALL"].iloc[0]
    return {
        "check": "Outlier flags (1.5 x IQR, within position group and metric)",
        "count": int(overall["n_flags"]),
        "treatment": (
            "Kept: they are real performances, and percentiles are rank-based and robust. "
            f"{overall['share_600_900']:.0%} of flags come from players with 600-900 minutes, "
            f"who are {overall['pool_share_600_900']:.0%} of the 600+ pool."
        ),
    }


def _spearman_matrix(players: pd.DataFrame) -> pd.DataFrame:
    """Return the Spearman matrix of the correlation metrics, labelled by metric name."""
    columns = {metric.season_column: metric.name for metric in CORRELATION_METRICS}
    return players[list(columns)].rename(columns=columns).corr(method="spearman")


def metric_correlation(
    conn: sqlite3.Connection, group: str | None = None, min_minutes: float = DEFAULT_MIN_MINUTES
) -> pd.DataFrame:
    """Return the Spearman correlation matrix of the metrics, in one group or pooled.

    It shows which metrics carry overlapping information. npxG stands in for xG,
    so xG is left out to avoid a near-duplicate. A pooled matrix also reflects
    differences between positions, which is why a group can be selected.
    """
    return _spearman_matrix(_season_rates(conn, min_minutes, group))


def metric_correlation_fig(
    conn: sqlite3.Connection, group: str | None = None, min_minutes: float = DEFAULT_MIN_MINUTES
) -> go.Figure:
    """Return a heatmap of the Spearman matrix, in one group or pooled.

    Blue and red are opposite poles around a neutral midpoint at zero, so the
    sign of a relationship reads at a glance. Exact values appear on hover.
    """
    matrix = metric_correlation(conn, group, min_minutes)
    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=list(matrix.columns),
            y=list(matrix.index),
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, DIVERGING_NEGATIVE], [0.5, DIVERGING_MIDPOINT], [1.0, DIVERGING_POSITIVE]],
            colorbar={"title": {"text": "Spearman r"}, "outlinewidth": 0},
            xgap=2,
            ygap=2,
            hovertemplate="%{y} and %{x}<br>r = %{z:.2f}<extra></extra>",
        )
    )
    title = f"Metric correlations: {group if group else 'all outfield players (pooled)'}"
    figure.update_layout(**_chart_layout(title, "", "", False), height=680)
    figure.update_xaxes(tickangle=-45, automargin=True)
    figure.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    return figure


def _within_group_correlation(rates: pd.DataFrame) -> pd.DataFrame:
    """Return the n-weighted average of within-group Spearman matrices.

    Groups with fewer than MIN_GROUP_PLAYERS players are left out as too noisy.
    """
    total, players_used = None, 0
    for _, players in rates.groupby("position_group"):
        if len(players) < MIN_GROUP_PLAYERS:
            continue
        weighted = _spearman_matrix(players) * len(players)
        total = weighted if total is None else total + weighted
        players_used += len(players)
    if total is None:
        raise ValueError(f"No position group has at least {MIN_GROUP_PLAYERS} players")
    return total / players_used


def top_correlated_pairs(
    conn: sqlite3.Connection, scope: str = "pooled", min_minutes: float = DEFAULT_MIN_MINUTES
) -> pd.DataFrame:
    """Return the 10 most positive and 5 most negative metric pairs for one scope.

    Pooling players from different positions can create correlations that are
    really about position: strikers shoot more and defend less than center
    backs. Each pair therefore shows its pooled r, its n-weighted average r within
    groups, and the difference. Scope "pooled" or "within_group" sets the ranking.
    """
    if scope not in SCOPE_COLUMNS:
        raise ValueError(f"scope must be one of {sorted(SCOPE_COLUMNS)}")
    rates = _season_rates(conn, min_minutes)
    pooled = _spearman_matrix(rates)
    within = _within_group_correlation(rates)
    pairs = pd.DataFrame(
        [
            {
                "metric_a": first,
                "metric_b": second,
                "pooled_r": pooled.loc[first, second],
                "within_group_r": within.loc[first, second],
            }
            for first, second in combinations(pooled.columns, 2)
        ]
    )
    pairs["pooled_minus_within"] = pairs["pooled_r"] - pairs["within_group_r"]
    column = SCOPE_COLUMNS[scope]
    ranked = pairs.sort_values(column, ascending=False)
    positive = ranked[ranked[column] > 0].head(10).assign(direction="positive")
    negative = ranked[ranked[column] < 0].tail(5).iloc[::-1].assign(direction="negative")
    columns = ["direction", "metric_a", "metric_b", "pooled_r", "within_group_r", "pooled_minus_within"]
    return pd.concat([positive, negative], ignore_index=True)[columns]


def _team_match_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return one row per team per match with goals, points and npxG for and against."""
    query = """
        WITH team_matches AS (
            SELECT match_id, home_team_id AS team_id, away_team_id AS opponent_id,
                   home_score AS goals_for, away_score AS goals_against
            FROM matches
            UNION ALL
            SELECT match_id, away_team_id, home_team_id, away_score, home_score
            FROM matches
        ),
        team_npxg AS (
            SELECT match_id, team_id, SUM(npxg) AS npxg
            FROM player_match
            GROUP BY match_id, team_id
        )
        SELECT
            tm.match_id,
            t.team_name AS team,
            tm.goals_for,
            tm.goals_against,
            CASE
                WHEN tm.goals_for > tm.goals_against THEN 3
                WHEN tm.goals_for = tm.goals_against THEN 1
                ELSE 0
            END AS points,
            ours.npxg AS npxg_for,
            theirs.npxg AS npxg_against
        FROM team_matches AS tm
        JOIN teams AS t ON t.team_id = tm.team_id
        JOIN team_npxg AS ours ON ours.match_id = tm.match_id AND ours.team_id = tm.team_id
        JOIN team_npxg AS theirs ON theirs.match_id = tm.match_id AND theirs.team_id = tm.opponent_id
    """
    return pd.read_sql_query(query, conn)


def team_season_table(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the league table with results, goals and non-penalty xG for each team.

    It connects player metrics to team results. npxG against is the opponent's
    sum in the same match. Teams are ordered by points, goal difference, then goals.
    """
    rows = _team_match_rows(conn)
    table = (
        rows.groupby("team")
        .agg(
            matches=("match_id", "count"),
            wins=("points", lambda points: int((points == 3).sum())),
            draws=("points", lambda points: int((points == 1).sum())),
            losses=("points", lambda points: int((points == 0).sum())),
            points=("points", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            npxg_for=("npxg_for", "sum"),
            npxg_against=("npxg_against", "sum"),
        )
        .reset_index()
    )
    table["goal_difference"] = table["goals_for"] - table["goals_against"]
    table["npxg_difference"] = table["npxg_for"] - table["npxg_against"]
    columns = [
        "team",
        "matches",
        "wins",
        "draws",
        "losses",
        "points",
        "goals_for",
        "goals_against",
        "goal_difference",
        "npxg_for",
        "npxg_against",
        "npxg_difference",
    ]
    return table.sort_values(
        ["points", "goal_difference", "goals_for"], ascending=False, ignore_index=True
    )[columns]


def team_season_checks(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return the validation checks on the team table as check, passed and detail.

    A league table has strict arithmetic, so these catch join errors: every team
    plays the same number of matches, each match awards 2 or 3 points, and goals
    and npxG balance between for and against. The order check tests that points
    broadly follow goal difference.
    """
    rows = _team_match_rows(conn)
    table = team_season_table(conn)
    wrong_matches = table.loc[table["matches"] != SEASON_MATCHES, "team"].tolist()
    points_per_match = rows.groupby("match_id")["points"].sum()
    wrong_points = int((~points_per_match.isin([2, 3])).sum())
    goal_gap = int(table["goals_for"].sum() - table["goals_against"].sum())
    npxg_gap = float(table["npxg_for"].sum() - table["npxg_against"].sum())
    player_npxg = conn.execute("SELECT SUM(npxg) FROM player_match").fetchone()[0]
    total_gap = float(table["npxg_for"].sum() - player_npxg)
    order_r = float(table[["points", "goal_difference"]].corr(method="spearman").iloc[0, 1])
    checks = [
        (
            f"Every team has {SEASON_MATCHES} matches",
            not wrong_matches,
            f"{len(table)} teams; teams with another count: {wrong_matches or 'none'}",
        ),
        (
            "Each match awards 2 or 3 points",
            wrong_points == 0,
            f"{len(points_per_match)} matches; {wrong_points} outside 2 or 3",
        ),
        ("Goals for equal goals against", goal_gap == 0, f"difference {goal_gap}"),
        ("npxG for equals npxG against", abs(npxg_gap) < TOTALS_TOLERANCE, f"difference {npxg_gap:.6f}"),
        (
            "Team npxG equals player_match npxG",
            abs(total_gap) < TOTALS_TOLERANCE,
            f"difference {total_gap:.6f}",
        ),
        (
            f"Points order follows goal difference (Spearman >= {PLAUSIBLE_ORDER_R})",
            order_r >= PLAUSIBLE_ORDER_R,
            f"Spearman {order_r:.3f}; top {table['team'].iloc[0]}, bottom {table['team'].iloc[-1]}",
        ),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def _correlation_row(label: str, x: pd.Series, y: pd.Series) -> dict:
    """Return Pearson and Spearman r and the least-squares line for two columns."""
    slope, intercept = np.polyfit(x, y, 1)
    return {
        "relationship": label,
        "n": len(x),
        "pearson_r": float(np.corrcoef(x, y)[0, 1]),
        "spearman_r": float(np.corrcoef(x.rank(), y.rank())[0, 1]),
        "slope": float(slope),
        "intercept": float(intercept),
    }


def team_level_correlations(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return team-level correlations of npxG difference with points and goal difference.

    They test whether chance quality tracks results. With only 12 teams these
    correlations are approximate, and a few teams can move them.
    """
    table = team_season_table(conn)
    return pd.DataFrame(
        [
            _correlation_row("npxg_difference vs points", table["npxg_difference"], table["points"]),
            _correlation_row(
                "npxg_difference vs goal_difference",
                table["npxg_difference"],
                table["goal_difference"],
            ),
        ]
    )


def _team_label_positions(table: pd.DataFrame) -> list[str]:
    """Return a label side per team that avoids collisions and the plot edges.

    Teams near either edge are labelled toward the center. The rest alternate
    sides in points order, so teams with similar points do not share a side.
    """
    x = table["npxg_difference"]
    edge = (x.max() - x.min()) / 4
    points_order = table["points"].rank(method="first", ascending=False).astype(int) - 1
    positions = []
    for value, order in zip(x, points_order):
        if value > x.max() - edge:
            positions.append("middle left")
        elif value < x.min() + edge:
            positions.append("middle right")
        else:
            positions.append("middle left" if order % 2 else "middle right")
    return positions


def team_npxg_vs_points_fig(conn: sqlite3.Connection) -> go.Figure:
    """Return a scatter of npxG difference against points, with a least-squares line.

    A team above the line earned more points than its chance difference suggests.
    """
    table = team_season_table(conn).sort_values("npxg_difference", ignore_index=True)
    fit = _correlation_row("fit", table["npxg_difference"], table["points"])
    line_x = np.array([table["npxg_difference"].min(), table["npxg_difference"].max()])
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=line_x,
            y=fit["slope"] * line_x + fit["intercept"],
            mode="lines",
            name="Least-squares line",
            line={"color": TEXT_SECONDARY, "width": 2, "dash": "dash"},
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=table["npxg_difference"],
            y=table["points"],
            mode="markers+text",
            name="Team",
            text=table["team"],
            textposition=_team_label_positions(table),
            textfont={"color": TEXT_SECONDARY, "size": 11},
            marker={"color": SERIES_COLORS[0], "size": 11, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="%{text}<br>npxG difference: %{x:.1f}<br>Points: %{y}<extra></extra>",
        )
    )
    figure.update_layout(
        **_chart_layout("Non-penalty xG difference and points", "npxG difference", "Points", True)
    )
    figure.update_xaxes(showgrid=True, gridcolor=GRID)
    figure.add_annotation(
        x=1,
        xref="paper",
        y=0,
        yref="paper",
        xanchor="right",
        yanchor="bottom",
        showarrow=False,
        text=f"n = {fit['n']}, Pearson r = {fit['pearson_r']:.2f}, Spearman r = {fit['spearman_r']:.2f}",
        font={"color": TEXT_SECONDARY, "size": 12},
    )
    return figure
