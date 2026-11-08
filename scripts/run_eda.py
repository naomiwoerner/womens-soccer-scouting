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
        print_section("Reliability (r_full): metric x threshold", stability_pivot(stability), index=True)
        print_section("Group and threshold combinations skipped (n < 8)", eda.stability_skipped_groups(connection))
        print_section("Median minutes per half, included players", eda.median_minutes_per_half(connection))
        if arguments.figures_dir:
            save_figures(connection, arguments.figures_dir)
            print(f"\nSaved figures to {arguments.figures_dir}")
    connection.close()


if __name__ == "__main__":
    main()
