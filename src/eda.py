"""Exploratory analysis of the scouting database.

Every function takes an open SQLite connection and returns a DataFrame or a
Plotly figure. Nothing here reads data/raw, so the app can run from the
database alone.
"""

import re
import sqlite3
from dataclasses import dataclass

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
