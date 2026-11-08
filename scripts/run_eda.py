"""Run the exploratory analysis and print its tables.

Run from the project root with: python scripts/run_eda.py [--figures-dir DIR]
With --figures-dir, the three figures are also saved there as HTML.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import eda  # noqa: E402
from src.metrics import METRICS  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
SLIDER_EXAMPLES = (300, 600, 900, 1200)
DESCRIPTIVE_EXAMPLES = ("npxg", "progressive_passes")


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
        if arguments.figures_dir:
            save_figures(connection, arguments.figures_dir)
            print(f"\nSaved figures to {arguments.figures_dir}")
    connection.close()


if __name__ == "__main__":
    main()
