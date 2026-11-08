"""Step 3: compute minutes played per player per match.

Minutes come from the stints in the lineup data, with these corrections:

- Stint times are read on the match clock, where the second half restarts at
  45:00. First-half stoppage time therefore overlaps early second-half times,
  so every time is converted to continuous match time using its period.
- A stint with no end time runs to the actual end of the match, taken from the
  Half End events, so stoppage time counts.
- Sent-off players can have stints recorded after their dismissal, and a second
  yellow for bad behaviour does not end a stint at all. Every stint is cut at
  the sending-off time from the player's card records.
- A player who returns from treatment and is then substituted can be left on
  the pitch until the final whistle. Every stint is also cut at the player's
  Substitution event.
- Temporary exits such as injury treatment are gaps between stints, so that time
  is not counted.

Run from the project root with: python -m src.minutes
"""

from collections import defaultdict

import pandas as pd

from src.data_loader import EVENTS_PATH, LINEUPS_PATH

# Where each period starts on the match clock, in seconds.
PERIOD_START_SECONDS = {1: 0, 2: 45 * 60, 3: 90 * 60, 4: 105 * 60}
SENDING_OFF_CARDS = {"Second Yellow", "Red Card"}
PLAYERS_PER_TEAM = 11


def period_lengths(events: pd.DataFrame) -> dict[int, dict[int, int]]:
    """Return {match_id: {period: length in seconds}} from the Half End events."""
    half_ends = events.loc[events["type"] == "Half End"]
    end_clock = (
        (half_ends["minute"] * 60 + half_ends["second"])
        .groupby([half_ends["match_id"], half_ends["period"]])
        .max()
    )
    lengths: dict[int, dict[int, int]] = defaultdict(dict)
    for (match_id, period), clock in end_clock.items():
        lengths[int(match_id)][int(period)] = int(clock) - PERIOD_START_SECONDS[int(period)]
    return dict(lengths)


def clock_to_seconds(clock: str) -> int:
    """Convert an "MM:SS" match-clock string to seconds."""
    minutes, seconds = clock.split(":")
    return int(minutes) * 60 + int(seconds)


def match_time(clock_seconds: int, period: int, lengths: dict[int, int]) -> int:
    """Convert a match-clock time in a period to seconds since kick-off."""
    in_period = clock_seconds - PERIOD_START_SECONDS[period]
    in_period = min(max(in_period, 0), lengths[period])
    return sum(length for p, length in lengths.items() if p < period) + in_period


def stint_interval(stint: dict, lengths: dict[int, int]) -> tuple[int, int]:
    """Return one stint's (start, end) in seconds since kick-off."""
    from_period = int(stint["from_period"])
    start = match_time(clock_to_seconds(stint["from"]), from_period, lengths)
    if pd.isna(stint["to"]):
        return start, sum(lengths.values())
    # A few stints are labelled as ending in an earlier period than they start.
    to_period = max(int(stint["to_period"]), from_period)
    return start, match_time(clock_to_seconds(stint["to"]), to_period, lengths)


def sending_off_time(cards, lengths: dict[int, int]) -> int | None:
    """Return when a player was sent off, in seconds since kick-off, if they were."""
    times = [
        match_time(clock_to_seconds(card["time"]), int(card["period"]), lengths)
        for card in cards
        if card["card_type"] in SENDING_OFF_CARDS
    ]
    return min(times) if times else None


def substitution_times(
    events: pd.DataFrame, lengths_by_match: dict[int, dict[int, int]]
) -> dict[tuple[int, int], int]:
    """Return {(match_id, player_id): seconds since kick-off} for players taken off."""
    substitutions = events.loc[events["type"] == "Substitution"]
    return {
        (int(event.match_id), int(event.player_id)): match_time(
            int(event.minute) * 60 + int(event.second),
            int(event.period),
            lengths_by_match[int(event.match_id)],
        )
        for event in substitutions.itertuples(index=False)
    }


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Combine touching or overlapping intervals so no second is counted twice."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def compute_minutes(lineups: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Return one row per player who was on the pitch in a match."""
    lengths_by_match = period_lengths(events)
    subbed_off_times = substitution_times(events, lengths_by_match)
    rows = []
    for squad_row in lineups.itertuples(index=False):
        if len(squad_row.positions) == 0:
            continue
        lengths = lengths_by_match[int(squad_row.match_id)]
        match_end = sum(lengths.values())

        intervals = [stint_interval(stint, lengths) for stint in squad_row.positions]
        sent_off_at = sending_off_time(squad_row.cards, lengths)
        subbed_off_at = subbed_off_times.get((int(squad_row.match_id), int(squad_row.player_id)))
        exits = [t for t in (sent_off_at, subbed_off_at) if t is not None]
        if exits:
            left_at = min(exits)
            intervals = [(s, min(e, left_at)) for s, e in intervals if s < left_at]
        intervals = merge_intervals([(s, e) for s, e in intervals if e > s])
        seconds = sum(end - start for start, end in intervals)
        if seconds == 0:
            continue

        rows.append(
            {
                "match_id": int(squad_row.match_id),
                "team_id": int(squad_row.team_id),
                "player_id": int(squad_row.player_id),
                "minutes": seconds / 60,
                "match_minutes": match_end / 60,
                "started": any(s["start_reason"] == "Starting XI" for s in squad_row.positions),
                "played_to_end": intervals[-1][1] == match_end,
                "sent_off_minute": None if sent_off_at is None else sent_off_at / 60,
            }
        )
    return pd.DataFrame(rows)


def load_minutes() -> pd.DataFrame:
    """Compute minutes from the Parquet files in data/raw/."""
    lineups = pd.read_parquet(LINEUPS_PATH)
    events = pd.read_parquet(
        EVENTS_PATH,
        columns=["match_id", "type", "period", "minute", "second", "player_id"],
        filters=[("type", "in", ["Half End", "Substitution"])],
    )
    return compute_minutes(lineups, events)


def team_match_totals(minutes: pd.DataFrame) -> pd.DataFrame:
    """Compare each team's total minutes with the total expected for that match.

    The expected total is 11 x match length, less the time the team played a
    player short because of sending-offs.
    """
    with_lost_time = minutes.assign(
        lost_to_red=(minutes["match_minutes"] - minutes["sent_off_minute"]).fillna(0)
    )
    totals = (
        with_lost_time.groupby(["match_id", "team_id"])
        .agg(
            total_minutes=("minutes", "sum"),
            match_minutes=("match_minutes", "first"),
            lost_to_red=("lost_to_red", "sum"),
        )
        .reset_index()
    )
    totals["expected_minutes"] = (
        PLAYERS_PER_TEAM * totals["match_minutes"] - totals["lost_to_red"]
    )
    totals["difference"] = totals["total_minutes"] - totals["expected_minutes"]
    return totals


def print_sanity_checks(minutes: pd.DataFrame) -> None:
    """Print the two sanity checks: full-match starters and team totals."""
    match_lengths = minutes.groupby("match_id")["match_minutes"].first()
    print(f"Player-match rows: {len(minutes):,} across {len(match_lengths)} matches")
    print(
        f"Match length: min {match_lengths.min():.1f}, "
        f"median {match_lengths.median():.1f}, max {match_lengths.max():.1f}"
    )

    full_match = minutes[
        minutes["started"] & minutes["played_to_end"] & minutes["sent_off_minute"].isna()
    ]
    print(f"\nStarters who finished the match: {len(full_match):,}")
    print(f"  Minutes: {full_match['minutes'].describe().round(1).to_dict()}")
    print(f"  Under 90 minutes: {(full_match['minutes'] < 90).sum()}")

    too_long = minutes[minutes["minutes"] > minutes["match_minutes"] + 1e-9]
    print(f"\nPlayers credited with more than the match length: {len(too_long)}")

    totals = team_match_totals(minutes)
    print(f"\nTeam-match totals: {len(totals)}")
    print(f"  Total minutes: {totals['total_minutes'].describe().round(1).to_dict()}")
    print(f"  Actual minus expected: {totals['difference'].describe().round(1).to_dict()}")
    print(f"  Teams with a sending-off: {(totals['lost_to_red'] > 0).sum()}")
    print("  Largest shortfalls against expected:")
    print(totals.nsmallest(5, "difference").round(1).to_string(index=False))
    print("  Largest excesses over expected:")
    print(totals.nlargest(3, "difference").round(1).to_string(index=False))


if __name__ == "__main__":
    print_sanity_checks(load_minutes())
