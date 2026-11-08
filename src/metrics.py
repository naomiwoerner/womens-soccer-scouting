"""Compute the per-player, per-match scouting metrics from StatsBomb events.

StatsBomb's pitch is 120 x 80 yards, and every event is recorded with the
acting team attacking toward x = 120, so forward distance is simply the change
in x.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

FINAL_THIRD_X = 80.0
OWN_THIRD_X = 40.0
PROGRESSIVE_YARDS = 10.0
SUCCESSFUL_OUTCOMES = {"Won", "Success In Play", "Success Out"}
AERIAL_WON_COLUMNS = [
    "pass_aerial_won",
    "shot_aerial_won",
    "clearance_aerial_won",
    "miscontrol_aerial_won",
]

EVENT_TYPES = [
    "Half End",
    "Substitution",
    "Shot",
    "Pressure",
    "Pass",
    "Carry",
    "Dribble",
    "Duel",
    "Interception",
    "Ball Recovery",
    "Clearance",
    "Miscontrol",
]
EVENT_COLUMNS = [
    "id",
    "match_id",
    "type",
    "period",
    "minute",
    "second",
    "player_id",
    "location",
    "shot_statsbomb_xg",
    "shot_type",
    "shot_key_pass_id",
    "pass_outcome",
    "pass_end_location",
    "pass_assisted_shot_id",
    "carry_end_location",
    "dribble_outcome",
    "duel_type",
    "duel_outcome",
    "interception_outcome",
    "ball_recovery_recovery_failure",
    *AERIAL_WON_COLUMNS,
]


@dataclass(frozen=True)
class Metric:
    """A scouting metric stored in player_match and described in metric_definitions."""

    name: str
    definition: str
    column_type: str = "INTEGER"
    higher_is_better: bool = True
    per_90: bool = True

    @property
    def season_column(self) -> str:
        """Return this metric's rate column in player_season_p90."""
        return f"{self.name}_p90" if self.per_90 else self.name


METRICS = [
    Metric("shots", "Shots taken, including penalties."),
    Metric(
        "xg",
        "Sum of StatsBomb expected goals (xG) on the player's shots, including penalties.",
        "REAL",
    ),
    Metric("np_shots", "Shots taken, excluding penalties."),
    Metric("npxg", "Sum of StatsBomb xG on the player's shots, excluding penalties.", "REAL"),
    Metric("xa", "Sum of the xG of the shots that the player's passes directly set up.", "REAL"),
    Metric("key_passes", "Passes that directly set up a shot, including goal assists."),
    Metric(
        "passes_completed",
        "Passes of any type, including set pieces, with no failure outcome recorded.",
    ),
    Metric(
        "pass_completion_pct",
        "Completed passes as a percentage of all pass attempts of any type. "
        "Attempts with an unknown outcome or kicked out for an injury count as not completed.",
        "REAL",
        per_90=False,
    ),
    Metric(
        "progressive_passes",
        "Completed passes that start at x >= 40 (outside the player's own defensive third) "
        "and end at least 10 yards closer to the opponent's goal line, measured along the "
        "length of the pitch.",
    ),
    Metric(
        "passes_into_final_third",
        "Completed passes that start before x = 80 and end beyond x = 80.",
    ),
    Metric(
        "progressive_carries",
        "Carries that start at x >= 40 (outside the player's own defensive third) and end "
        "at least 10 yards closer to the opponent's goal line, measured along the length "
        "of the pitch.",
    ),
    Metric("successful_dribbles", "Dribbles with a Complete outcome."),
    Metric("pressures", "Pressure events applied to an opponent on the ball."),
    Metric(
        "tackles_interceptions",
        "Tackles won plus successful interceptions: tackle duels and interceptions with "
        "an outcome of Won, Success In Play or Success Out.",
    ),
    Metric("ball_recoveries", "Ball recoveries, excluding those marked as failed."),
    Metric(
        "aerial_duels_won",
        "Aerial duels won, taken from the aerial-won flag on passes, shots, clearances "
        "and miscontrols.",
    ),
]


def _x_coordinates(locations: pd.Series, mask: pd.Series) -> pd.Series:
    """Return the x coordinate of each location where mask is true, NaN elsewhere."""
    x = pd.Series(np.nan, index=locations.index, dtype=float)
    x[mask] = [location[0] for location in locations[mask]]
    return x


def _is_progressive(start_x: pd.Series, end_x: pd.Series) -> pd.Series:
    """Return True where the ball moved far enough toward the opponent's goal.

    Coordinates have one decimal place, so the difference is rounded to stop
    floating-point error from rejecting moves of exactly 10 yards.
    """
    return (end_x - start_x).round(1) >= PROGRESSIVE_YARDS


def player_match_stats(events: pd.DataFrame) -> pd.DataFrame:
    """Return one row of metric counts per player per match."""
    player_events = events[events["player_id"].notna()]
    event_type = player_events["type"]

    is_shot = event_type == "Shot"
    is_penalty = player_events["shot_type"] == "Penalty"
    is_pass = event_type == "Pass"
    is_carry = event_type == "Carry"
    is_complete_pass = is_pass & player_events["pass_outcome"].isna()

    shot_xg = player_events["shot_statsbomb_xg"].where(is_shot, 0.0)
    xg_by_shot_id = player_events.loc[is_shot].set_index("id")["shot_statsbomb_xg"]

    pass_start = _x_coordinates(player_events["location"], is_pass)
    pass_end = _x_coordinates(player_events["pass_end_location"], is_pass)
    carry_start = _x_coordinates(player_events["location"], is_carry)
    carry_end = _x_coordinates(player_events["carry_end_location"], is_carry)

    is_tackle_won = (event_type == "Duel") & (player_events["duel_type"] == "Tackle") & player_events[
        "duel_outcome"
    ].isin(SUCCESSFUL_OUTCOMES)
    is_interception_won = (event_type == "Interception") & player_events[
        "interception_outcome"
    ].isin(SUCCESSFUL_OUTCOMES)

    stats = player_events[["match_id", "player_id"]].assign(
        shots=is_shot,
        xg=shot_xg,
        np_shots=is_shot & ~is_penalty,
        npxg=shot_xg.where(~is_penalty, 0.0),
        xa=player_events["pass_assisted_shot_id"].map(xg_by_shot_id).fillna(0.0),
        key_passes=player_events["pass_assisted_shot_id"].notna(),
        passes_attempted=is_pass,
        passes_completed=is_complete_pass,
        progressive_passes=is_complete_pass
        & (pass_start >= OWN_THIRD_X)
        & _is_progressive(pass_start, pass_end),
        passes_into_final_third=is_complete_pass
        & (pass_start < FINAL_THIRD_X)
        & (pass_end > FINAL_THIRD_X),
        progressive_carries=is_carry
        & (carry_start >= OWN_THIRD_X)
        & _is_progressive(carry_start, carry_end),
        successful_dribbles=(event_type == "Dribble")
        & (player_events["dribble_outcome"] == "Complete"),
        pressures=event_type == "Pressure",
        tackles_interceptions=is_tackle_won | is_interception_won,
        ball_recoveries=(event_type == "Ball Recovery")
        & ~player_events["ball_recovery_recovery_failure"].eq(True),
        aerial_duels_won=player_events[AERIAL_WON_COLUMNS].eq(True).any(axis=1),
    )
    return stats.groupby(["match_id", "player_id"], as_index=False).sum()


def build_player_match(minutes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per player per match with minutes and every metric."""
    stats = player_match_stats(events)
    stats["player_id"] = stats["player_id"].astype(int)

    player_match = minutes[["player_id", "match_id", "team_id", "minutes"]].merge(
        stats, on=["match_id", "player_id"], how="left", validate="one_to_one"
    )
    count_columns = [c for c in stats.columns if c not in ("match_id", "player_id")]
    player_match[count_columns] = player_match[count_columns].fillna(0)

    real_columns = {m.name for m in METRICS if m.column_type == "REAL"}
    integer_columns = [c for c in count_columns if c not in real_columns]
    player_match[integer_columns] = player_match[integer_columns].astype(int)

    # A player who attempted no passes has no completion percentage (stored as NULL).
    attempted = player_match["passes_attempted"]
    player_match["pass_completion_pct"] = (
        100 * player_match["passes_completed"] / attempted.where(attempted > 0)
    )
    return player_match
