"""Step 2: download every match's events and lineups and save them as Parquet.

Writes matches.parquet, events.parquet and lineups.parquet to data/raw/.
Run from the project root with: python -m src.data_loader [--force]
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from statsbombpy import sb

from src.competition import load_matches

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MATCHES_PATH = RAW_DIR / "matches.parquet"
EVENTS_PATH = RAW_DIR / "events.parquet"
LINEUPS_PATH = RAW_DIR / "lineups.parquet"

MAX_WORKERS = 8


def fetch_events(match_id: int) -> pd.DataFrame:
    """Return all events for one match."""
    return sb.events(match_id=match_id)


def fetch_lineups(match: pd.Series) -> pd.DataFrame:
    """Return one row per squad player for one match, tagged with match and team IDs.

    The `cards` and `positions` columns stay nested, as lists of dicts.
    """
    team_ids = {
        match["home_team"]: int(match["home_team_id"]),
        match["away_team"]: int(match["away_team_id"]),
    }
    lineups = sb.lineups(match_id=int(match["match_id"]))
    if set(lineups) != set(team_ids):
        raise ValueError(
            f"Match {match['match_id']}: lineup teams {sorted(lineups)} "
            f"do not match fixture teams {sorted(team_ids)}"
        )

    frames = [
        squad.assign(
            match_id=int(match["match_id"]),
            team_id=team_ids[team],
            team_name=team,
        )
        for team, squad in lineups.items()
    ]
    return pd.concat(frames, ignore_index=True)


def download_all(force: bool = False) -> None:
    """Download the whole season, skipping the work if the files already exist."""
    paths = (MATCHES_PATH, EVENTS_PATH, LINEUPS_PATH)
    if not force and all(path.exists() for path in paths):
        print(f"Raw files already exist in {RAW_DIR}. Use --force to re-download.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    matches = load_matches()
    match_rows = [row for _, row in matches.iterrows()]
    match_ids = [int(row["match_id"]) for row in match_rows]

    print(f"Downloading events and lineups for {len(match_ids)} matches...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        event_frames = list(pool.map(fetch_events, match_ids))
        lineup_frames = list(pool.map(fetch_lineups, match_rows))

    # Everything is written after the downloads finish, so a failed run
    # cannot leave a partial season on disk.
    events = pd.concat(event_frames, ignore_index=True, sort=False)
    lineups = pd.concat(lineup_frames, ignore_index=True, sort=False)

    matches.to_parquet(MATCHES_PATH, index=False)
    events.to_parquet(EVENTS_PATH, index=False)
    lineups.to_parquet(LINEUPS_PATH, index=False)

    for name, frame, path in (
        ("matches", matches, MATCHES_PATH),
        ("events", events, EVENTS_PATH),
        ("lineups", lineups, LINEUPS_PATH),
    ):
        size_mb = path.stat().st_size / 1_000_000
        print(f"  {name}: {len(frame):,} rows, {size_mb:.1f} MB -> {path.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-download existing files")
    download_all(force=parser.parse_args().force)
