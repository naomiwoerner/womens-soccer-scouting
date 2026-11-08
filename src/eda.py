"""Exploratory analysis of the scouting database.

Every function takes an open SQLite connection and returns a DataFrame or a
Plotly figure. Nothing here reads data/raw, so the app can run from the
database alone.
"""

import re
import sqlite3

import pandas as pd
import plotly.graph_objects as go

from src.metrics import METRICS

DEFAULT_THRESHOLDS = (300, 400, 600, 900, 1200)
MIN_GROUP_PLAYERS = 8
COMPLETION_METRIC = "pass_completion_pct"
FIRST = "first"
SECOND = "second"
HISTOGRAM_BIN_MINUTES = 100

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
CONTEXT_LINE = "#c3c2b7"
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


def _half_rates(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return one row per outfield player with minutes and all metrics for each half.

    Columns are (name, half) pairs, for example ("xg", "first"). A player absent
    from a half has missing values there. Completion % comes from that half's
    totals, and counts are converted to per-90 rates using that half's minutes.
    """
    per_90_metrics = [metric.name for metric in METRICS if metric.per_90]
    sums = ",\n            ".join(f"SUM(pm.{name}) AS {name}" for name in per_90_metrics)
    query = f"""
        SELECT
            pm.player_id,
            p.position_group,
            CASE WHEN m.match_week <= ? THEN '{FIRST}' ELSE '{SECOND}' END AS half,
            SUM(pm.minutes) AS minutes,
            SUM(pm.passes_attempted) AS passes_attempted,
            {sums}
        FROM player_match AS pm
        JOIN players AS p ON p.player_id = pm.player_id
        JOIN matches AS m ON m.match_id = pm.match_id
        WHERE p.position_group IS NOT NULL
        GROUP BY pm.player_id, p.position_group, half
    """
    totals = pd.read_sql_query(query, conn, params=(_last_week_of_first_half(conn),))

    rates = totals[["player_id", "position_group", "half", "minutes"]].copy()
    for name in per_90_metrics:
        rates[name] = totals[name] * 90 / totals["minutes"]
    attempted = totals["passes_attempted"]
    rates[COMPLETION_METRIC] = 100 * totals["passes_completed"] / attempted.where(attempted > 0)
    return rates.pivot(index=["player_id", "position_group"], columns="half")


def _included_groups(
    halves: pd.DataFrame, threshold: int
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Return the groups analysed at a threshold, and the groups skipped for small n.

    A player needs at least half the threshold in each half of the season.
    """
    half_minimum = threshold / 2
    eligible = halves[
        (halves[("minutes", FIRST)] >= half_minimum)
        & (halves[("minutes", SECOND)] >= half_minimum)
    ]
    included, skipped = {}, []
    for group, players in eligible.groupby(level="position_group"):
        if len(players) < MIN_GROUP_PLAYERS:
            skipped.append({"threshold": threshold, "position_group": group, "n_players": len(players)})
        else:
            included[group] = players
    return included, skipped


def _weighted_split_correlation(groups: dict[str, pd.DataFrame], metric: str) -> tuple[float, int]:
    """Return a metric's n-weighted within-group Spearman correlation and total n.

    Correlating within groups stops the gap between positions (strikers shoot
    more than center backs in both halves) from inflating the result. Spearman
    is computed as the Pearson correlation of average ranks, which handles ties.
    """
    weighted_sum, total_n = 0.0, 0
    for players in groups.values():
        pair = players[[(metric, FIRST), (metric, SECOND)]].dropna()
        ranks = pair.rank()
        correlation = ranks[(metric, FIRST)].corr(ranks[(metric, SECOND)])
        weighted_sum += correlation * len(pair)
        total_n += len(pair)
    return (weighted_sum / total_n if total_n else float("nan")), total_n


def _spearman_brown(r_split: float) -> float:
    """Return the Spearman-Brown estimate 2r / (1 + r), undefined at r = -1."""
    return 2 * r_split / (1 + r_split) if r_split > -1 else float("nan")


def stability_by_threshold(
    conn: sqlite3.Connection, thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Return the split-half stability of every metric at each minutes threshold.

    A metric only helps a scout if it measures something repeatable about the
    player. For each threshold T, this takes outfield players with at least T/2
    minutes in each half and correlates their per-90 values between halves.
    The long result has threshold, metric, n_players, r_split and r_full, where
    r_full = 2r / (1 + r) estimates the reliability of a full-season sample.

    r_full describes the players actually included, and most of them exceed T/2
    in each half by a wide margin, so see median_minutes_per_half when reading it.
    """
    halves = _half_rates(conn)
    rows = []
    for threshold in thresholds:
        groups, _ = _included_groups(halves, threshold)
        for metric in METRICS:
            r_split, n_players = _weighted_split_correlation(groups, metric.name)
            rows.append(
                {
                    "threshold": threshold,
                    "metric": metric.name,
                    "n_players": n_players,
                    "r_split": r_split,
                    "r_full": _spearman_brown(r_split),
                }
            )
    return pd.DataFrame(rows)


def stability_skipped_groups(
    conn: sqlite3.Connection, thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Return the group and threshold combinations left out for having n < 8.

    A rank correlation on so few players is mostly noise. Recording what was
    dropped shows which position groups the stability averages do not cover.
    """
    halves = _half_rates(conn)
    skipped = [row for t in thresholds for row in _included_groups(halves, t)[1]]
    return pd.DataFrame(skipped, columns=["threshold", "position_group", "n_players"])


def median_minutes_per_half(
    conn: sqlite3.Connection, thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Return the median minutes per half among the players in the stability sample.

    The threshold only sets a floor, so this shows how much data the included
    players really have. It is the context for reading r_full: when the median
    is far above T/2, r_full reflects that larger sample rather than T minutes.
    """
    halves = _half_rates(conn)
    rows = []
    for threshold in thresholds:
        groups, _ = _included_groups(halves, threshold)
        minutes = pd.concat(groups.values())["minutes"]
        rows.append(
            {
                "threshold": threshold,
                "half_minimum": threshold / 2,
                "n_players": len(minutes),
                "median_first_half": minutes[FIRST].median(),
                "median_second_half": minutes[SECOND].median(),
                "median_per_half": pd.concat([minutes[FIRST], minutes[SECOND]]).median(),
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
    conn: sqlite3.Connection, thresholds: tuple[int, ...] = DEFAULT_THRESHOLDS
) -> go.Figure:
    """Return r_full against threshold, one faint line per metric and a bold mean.

    It shows whether a higher minimum buys meaningfully better reliability. The
    metric lines share one quiet color and name themselves on hover, because 16
    separate hues could not be told apart.
    """
    stability = stability_by_threshold(conn, thresholds)
    figure = go.Figure()
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

    mean = stability.groupby("threshold", as_index=False)["r_full"].mean()
    figure.add_trace(
        go.Scatter(
            x=mean["threshold"],
            y=mean["r_full"],
            mode="lines+markers",
            name="Mean of all metrics",
            line={"color": SERIES_COLORS[0], "width": 3},
            marker={"color": SERIES_COLORS[0], "size": 9, "line": {"color": SURFACE, "width": 2}},
            hovertemplate="Mean<br>T = %{x}<br>r_full = %{y:.2f}<extra></extra>",
        )
    )
    figure.add_annotation(
        x=mean["threshold"].iloc[-1],
        y=mean["r_full"].iloc[-1],
        text="Mean",
        showarrow=False,
        xanchor="left",
        xshift=10,
        font={"color": TEXT_PRIMARY, "size": 12},
    )
    figure.update_layout(
        **_chart_layout(
            "Reliability by minutes threshold", "Minutes threshold (T)", "Reliability (r_full)", True
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
