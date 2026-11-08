"""Run the exploratory analysis and print its tables.

Run from the project root with: python scripts/run_eda.py [--figures-dir DIR]
With --figures-dir, the three figures are also saved there as HTML.
"""

import argparse
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import eda  # noqa: E402
from src.metrics import METRICS  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
SLIDER_EXAMPLES = (300, 600, 900, 1200)
DESCRIPTIVE_EXAMPLES = ("npxg", "progressive_passes")
NUMBER_PATTERN = re.compile(r"(?<![\w.])-?\d+(?:,\d{3})*(?:\.\d+)?%?")


def stability_pivot(stability: pd.DataFrame) -> pd.DataFrame:
    """Return r_full as metric x threshold, with mean and n_players rows, as text."""
    pivot = stability.pivot(index="metric", columns="threshold", values="r_full")
    pivot = pivot.reindex([metric.name for metric in METRICS])
    table = pivot.map(lambda value: f"{value:.2f}")
    table.loc["MEAN"] = [f"{value:.2f}" for value in pivot.mean()]
    table.loc["n_players"] = [
        str(value) for value in stability.groupby("threshold")["n_players"].max()
    ]
    return table


def stability_range_pivot(stability: pd.DataFrame) -> pd.DataFrame:
    """Return each metric's 10th to 90th percentile of r_full as metric x threshold text."""
    ranges = stability.assign(
        r_full_range=[
            f"{low:.2f}-{high:.2f}"
            for low, high in zip(stability["r_full_p10"], stability["r_full_p90"])
        ]
    )
    pivot = ranges.pivot(index="metric", columns="threshold", values="r_full_range")
    return pivot.reindex([metric.name for metric in METRICS])


def within_group_r(connection: sqlite3.Connection, first: str, second: str, min_minutes: float) -> float:
    """Return a pair's n-weighted within-group r, built from each group's own matrix."""
    sizes = eda.descriptive_stats(connection, min_minutes=min_minutes).drop_duplicates("position_group")
    total, players = 0.0, 0
    for group, n_players in zip(sizes["position_group"], sizes["n"]):
        if n_players >= eda.MIN_GROUP_PLAYERS:
            total += eda.metric_correlation(connection, group, min_minutes).loc[first, second] * n_players
            players += n_players
    return float(total / players)


def expected_insight_numbers(connection: sqlite3.Connection, min_minutes: float) -> dict[str, list[str]]:
    """Return every number each insight should contain, recomputed from its source function."""
    stats = eda.descriptive_stats(connection, min_minutes=min_minutes)

    def median(group: str, metric: str) -> float:
        rows = stats[(stats["position_group"] == group) & (stats["metric"] == metric)]
        return float(rows["median"].iloc[0])

    def pooled_r(first: str, second: str) -> float:
        return float(eda.metric_correlation(connection, None, min_minutes).loc[first, second])

    def slider(minutes: float) -> float:
        return float(eda.reliability_at_minutes(connection, minutes)["mean_r_full"].iloc[0])

    means = eda.stability_mean_range(connection).sort_values("threshold")
    cutoff = eda.COMPARISON_CUTOFF
    threshold = [
        str(means["threshold"].iloc[-2]),
        str(means["threshold"].iloc[-1]),
        f"{means['mean_r_full'].iloc[-1] - means['mean_r_full'].iloc[-2]:.2f}",
        f"{slider(min_minutes):.2f}",
        f"{min_minutes:g}",
    ]
    if min_minutes < cutoff:
        table = eda.minutes_threshold_table(connection, (min_minutes, cutoff)).set_index("threshold")
        threshold += [
            f"{slider(cutoff):.2f}",
            str(cutoff),
            str(cutoff),
            str(table.loc[min_minutes, "total"] - table.loc[cutoff, "total"]),
            str(table.loc[min_minutes, "total"]),
            str(table.loc[min_minutes, "Striker"] - table.loc[cutoff, "Striker"]),
            str(table.loc[min_minutes, "Striker"]),
        ]
    else:
        threshold.append(str(cutoff))

    tier_threshold = min(min_minutes, 2 * eda.POOL_HALF_MINUTES)
    tiers = eda.metric_reliability(connection, tier_threshold)
    actual = 2 * eda.median_minutes_per_half(connection, (tier_threshold,))["median_per_half"].iloc[0]

    progressive = stats[stats["metric"] == "progressive_passes"].nlargest(2, "median")["median"]

    teams = eda.team_season_table(connection).sort_values("npxg_difference", ignore_index=True)
    slope, intercept = np.polyfit(teams["npxg_difference"], teams["goal_difference"], 1)
    fitted = slope * teams["npxg_difference"] + intercept
    gaps = teams["goal_difference"] - fitted
    worst = gaps.abs().idxmax()
    extremes = eda.TEAM_EXTREMES
    middle = teams.iloc[extremes:-extremes]

    _, outlier_summary = eda.outlier_table(connection)
    overall = outlier_summary[outlier_summary["metric"] == "ALL"].iloc[0]
    minimum, band = eda.DEFAULT_MIN_MINUTES, eda.OUTLIER_MINUTES_BAND

    return {
        "threshold": threshold,
        "reliability_tiers": [
            f"{tier_threshold:g}",
            f"{actual:.0f}",
            f"{eda.HIGH_RELIABILITY:.2f}",
            f"{eda.MODERATE_RELIABILITY:.2f}",
            str(int((tiers["tier"] == "moderate").sum())),
        ],
        "position_profiles": [
            f"{median('Striker', 'npxg'):.2f}",
            f"{median('Center Back', 'npxg'):.2f}",
            *[f"{value:.2f}" for value in progressive],
        ],
        "confounding": [
            f"{pooled_r('np_shots', 'passes_completed'):.2f}",
            f"{within_group_r(connection, 'np_shots', 'passes_completed', min_minutes):.2f}",
            f"{median('Striker', 'np_shots'):.1f}",
            f"{median('Striker', 'passes_completed'):.1f}",
            f"{median('Center Back', 'np_shots'):.1f}",
            f"{median('Center Back', 'passes_completed'):.1f}",
        ],
        "pooling_hides_links": [
            f"{pooled_r('key_passes', 'passes_into_final_third'):.2f}",
            f"{within_group_r(connection, 'key_passes', 'passes_into_final_third', min_minutes):.2f}",
        ],
        "pressing_vs_passing": [
            f"{within_group_r(connection, 'passes_completed', 'pressures', min_minutes):.2f}"
        ],
        "team_level": [
            str(len(teams)),
            f"{np.corrcoef(teams['npxg_difference'], teams['points'])[0, 1]:.2f}",
            str(teams["goal_difference"][worst]),
            f"{fitted[worst]:.1f}",
            f"{abs(gaps[worst]):.1f}",
            str(len(teams)),
            str(extremes),
            str(extremes),
            f"{np.corrcoef(middle['npxg_difference'], middle['points'])[0, 1]:.2f}",
        ],
        "outliers": [
            str(int(overall["n_flags"])),
            str(eda.OUTLIER_IQR_MULTIPLE),
            str(minimum),
            str(minimum),
            str(band),
            f"{overall['share_600_900']:.0%}",
            f"{overall['pool_share_600_900']:.0%}",
        ],
    }


def insight_numbers(text: str, team_names: list[str]) -> list[str]:
    """Return the numbers in an insight, ignoring digits in team names and 'per 90'."""
    for name in team_names:
        text = text.replace(name, "")
    return NUMBER_PATTERN.findall(text.replace("per 90", ""))


def check_insights(connection: sqlite3.Connection, insights: list[dict], min_minutes: float) -> pd.DataFrame:
    """Return, per insight, whether its numbers match those recomputed from the source functions."""
    expected = expected_insight_numbers(connection, min_minutes)
    team_names = eda.team_season_table(connection)["team"].tolist()
    rows = []
    for insight in insights:
        found = Counter(insight_numbers(insight["text"], team_names))
        wanted = Counter(expected[insight["id"]])
        rows.append(
            {
                "id": insight["id"],
                "numbers": sum(wanted.values()),
                "passed": found == wanted,
                "unexpected": sorted((found - wanted).elements()),
                "missing": sorted((wanted - found).elements()),
            }
        )
    return pd.DataFrame(rows)


def print_section(title: str, table: pd.DataFrame, index: bool = False) -> None:
    """Print a titled table."""
    print(f"\n=== {title} ===")
    print("None" if table.empty else table.to_string(index=index))


def save_figures(connection: sqlite3.Connection, directory: Path) -> None:
    """Save the three figures as HTML files."""
    directory.mkdir(parents=True, exist_ok=True)
    figures = {
        "minutes_histogram": eda.minutes_histogram_figure(connection),
        "reliability": eda.reliability_figure(connection),
        "qualifying_players": eda.qualifying_players_figure(connection),
    }
    for name, figure in figures.items():
        figure.write_html(directory / f"{name}.html", include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, help="save the figures as HTML here")
    arguments = parser.parse_args()

    pd.set_option("display.width", 250, "display.max_colwidth", None, "display.max_rows", None)
    with sqlite3.connect(DB_PATH) as connection:
        stability = eda.stability_by_threshold(connection)
        print_section("Data quality", eda.data_quality_summary(connection))
        print_section("Qualifying outfield players by threshold", eda.minutes_threshold_table(connection))
        print_section("Season split", eda.season_split(connection))
        print_section("Stability pool: 600+ minutes in each half", eda.stability_pool_sizes(connection))
        print_section("Position groups skipped (n < 8)", eda.stability_skipped_groups(connection))
        print_section("Reliability at about T minutes: mean r_full", stability_pivot(stability), index=True)
        print_section("10-90% of draws, per metric", stability_range_pivot(stability), index=True)
        print_section("Mean across metrics, 10-90% of draws", eda.stability_mean_range(connection).round(3))
        print_section("Median actual minutes per half", eda.median_minutes_per_half(connection).round(1))
        print_section("Metric reliability at 600 minutes", eda.metric_reliability(connection).round(3))
        by_minutes = pd.concat(
            [eda.reliability_at_minutes(connection, minutes) for minutes in SLIDER_EXAMPLES],
            ignore_index=True,
        )
        print_section("Reliability at minutes", by_minutes.round(3))

        descriptive = eda.descriptive_stats(connection)
        for metric in DESCRIPTIVE_EXAMPLES:
            print_section(f"Descriptive stats: {metric}", descriptive[descriptive["metric"] == metric].round(3))
        _, outlier_summary = eda.outlier_table(connection)
        print_section("Outlier summary", outlier_summary.round(3))
        print_section("Top correlated pairs: pooled", eda.top_correlated_pairs(connection, "pooled").round(3))
        print_section(
            "Top correlated pairs: averaged within group",
            eda.top_correlated_pairs(connection, "within_group").round(3),
        )
        print_section("Team season table", eda.team_season_table(connection).round(2))
        print_section("Team table checks", eda.team_season_checks(connection))
        print_section("Team-level correlations", eda.team_level_correlations(connection).round(3))

        insights = eda.eda_insights(connection)
        print("\n=== Insights ===")
        for insight in insights:
            print(f"\n[{insight['id']}] {insight['section']}: {insight['title']}")
            print(insight["text"])
        insight_check = check_insights(connection, insights, eda.DEFAULT_MIN_MINUTES)
        print_section("Insight number check", insight_check)
        if arguments.figures_dir:
            save_figures(connection, arguments.figures_dir)
            print(f"\nSaved figures to {arguments.figures_dir}")
    connection.close()

    if not insight_check["passed"].all():
        sys.exit("Insight check failed: see the lines marked False above.")


if __name__ == "__main__":
    main()
