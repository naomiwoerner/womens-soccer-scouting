"""Assign each player a primary position group from their lineup stints.

Each StatsBomb position belongs to one of five groups, or to goalkeeper. A
player's group is the one with the most minutes summed across its positions,
using the same corrected stints as the minutes calculation. Their primary
position is the most-played position within that group. Goalkeepers get no group.
"""

import pandas as pd

from src.minutes import period_lengths, played_stints, substitution_times

GOALKEEPER = "Goalkeeper"
CENTER_BACK = "Center Back"
FULL_BACK = "Full Back/Wing Back"
MIDFIELD = "Defensive/Central Midfield"
ATTACKING = "Attacking Midfield/Winger"
STRIKER = "Striker"

POSITION_GROUPS = {
    GOALKEEPER: GOALKEEPER,
    "Right Center Back": CENTER_BACK,
    "Center Back": CENTER_BACK,
    "Left Center Back": CENTER_BACK,
    "Right Back": FULL_BACK,
    "Left Back": FULL_BACK,
    "Right Wing Back": FULL_BACK,
    "Left Wing Back": FULL_BACK,
    "Right Defensive Midfield": MIDFIELD,
    "Center Defensive Midfield": MIDFIELD,
    "Left Defensive Midfield": MIDFIELD,
    "Right Center Midfield": MIDFIELD,
    "Center Midfield": MIDFIELD,
    "Left Center Midfield": MIDFIELD,
    "Right Midfield": ATTACKING,
    "Left Midfield": ATTACKING,
    "Right Attacking Midfield": ATTACKING,
    "Center Attacking Midfield": ATTACKING,
    "Left Attacking Midfield": ATTACKING,
    "Right Wing": ATTACKING,
    "Left Wing": ATTACKING,
    "Right Center Forward": STRIKER,
    "Center Forward": STRIKER,
    "Left Center Forward": STRIKER,
    "Secondary Striker": STRIKER,
}


def position_minutes(lineups: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return minutes per player, team and position across the season."""
    lengths_by_match = period_lengths(events)
    subbed_off_times = substitution_times(events, lengths_by_match)
    rows = []
    for squad_row in lineups.itertuples(index=False):
        if len(squad_row.positions) == 0:
            continue
        lengths = lengths_by_match[int(squad_row.match_id)]
        stints, _ = played_stints(squad_row, lengths, subbed_off_times)
        rows.extend(
            {
                "player_id": int(squad_row.player_id),
                "team_id": int(squad_row.team_id),
                "position": position,
                "minutes": (end - start) / 60,
            }
            for start, end, position in stints
        )
    return (
        pd.DataFrame(rows)
        .groupby(["player_id", "team_id", "position"], as_index=False)["minutes"]
        .sum()
    )


def _most_minutes(frame: pd.DataFrame, keys: list[str], column: str) -> pd.DataFrame:
    """Return the value of column with the most minutes for each key.

    Ties are broken alphabetically so the result is repeatable.
    """
    totals = frame.groupby([*keys, column], as_index=False)["minutes"].sum()
    return totals.sort_values(
        [*keys, "minutes", column], ascending=[True] * len(keys) + [False, True]
    ).drop_duplicates(keys)[[*keys, column]]


def primary_positions(by_position: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Return each key's position group and most-played position within that group."""
    with_groups = by_position.assign(
        group=by_position["position"].map(POSITION_GROUPS.__getitem__)
    )
    top_groups = _most_minutes(with_groups, keys, "group")
    within_group = with_groups.merge(top_groups, on=[*keys, "group"])
    top_positions = _most_minutes(within_group, keys, "position")

    primary = top_groups.merge(top_positions, on=keys, validate="one_to_one")
    primary["position_group"] = primary["group"].where(primary["group"] != GOALKEEPER)
    return primary.rename(columns={"position": "primary_position"})[
        [*keys, "primary_position", "position_group"]
    ]
