"""Step 1: identify the competition and list its matches.

Run from the project root with: python -m src.competition
"""

import warnings

import pandas as pd
from statsbombpy import sb
from statsbombpy.api_client import NoAuthWarning

COMPETITION_ID = 135  # Frauen Bundesliga
SEASON_ID = 281  # 2023/2024

# Open data needs no credentials, so statsbombpy's reminder about that is noise.
warnings.filterwarnings("ignore", category=NoAuthWarning)


def load_matches() -> pd.DataFrame:
    """Return one row per match in the season, ordered by date."""
    matches = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    return matches.sort_values(["match_date", "match_id"]).reset_index(drop=True)


if __name__ == "__main__":
    matches = load_matches()
    print(f"Frauen Bundesliga 2023/24: {len(matches)} matches")
    print(f"  Match weeks: {matches['match_week'].nunique()}")
    print(f"  Dates: {matches['match_date'].min()} to {matches['match_date'].max()}")
    print(f"  Available: {(matches['match_status'] == 'available').sum()}")
