"""Check the report's display helpers and fact sheets against the database.

1. rank_phrase / players_ranked_below: for every player and every allowed
   metric, the count derived from PERCENT_RANK must equal the number of
   players in her group with a strictly lower raw value, counted here in
   Python from player_season_p90.
2. fit_label at and around each cutoff.
3. build_fact_sheet for every player in the reference pool, under a
   requirement for her group: formatted numbers, shortlist rank, strengths,
   weaknesses and caveats, each recomputed independently.

Run from the project root with: python scripts/run_report_check.py
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import fit  # noqa: E402
from src.eda import metric_reliability  # noqa: E402
from src.report import build_fact_sheet  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
# One requirement per group: the striker example from try_report plus a simple profile.
REQUIREMENTS = {
    "Striker": {"pressures": 5, "npxg": 4, "np_shots": 4, "ball_recoveries": 3},
    "Center Back": {"aerial_duels_won": 2, "progressive_passes": 1, "pass_completion_pct": 1},
    "Full Back/Wing Back": {"progressive_carries": 1, "xa": 1, "tackles_interceptions": 1},
    "Defensive/Central Midfield": {"passes_completed": 1, "ball_recoveries": 1},
    "Attacking Midfield/Winger": {"xa": 1, "key_passes": 1, "successful_dribbles": 1},
}

failures: list[str] = []


def fail(message: str) -> None:
    failures.append(message)
    if len(failures) <= 20:
        print(f"  FAIL {message}")


def expected_phrase(below: int, n: int, label: str) -> str:
    """The wording rule, written out again independently of src/fit.py."""
    if below == n - 1:
        return f"best of {n} {label}"
    if below == 0:
        return f"lowest of {n} {label}"
    return f"better than {below} of {n - 1} other {label}"


def check_rank_phrases(conn: sqlite3.Connection) -> None:
    print("1. Rank phrases against raw values")
    season = pd.read_sql_query("SELECT * FROM player_season_p90", conn)
    pctl = pd.read_sql_query("SELECT * FROM player_percentiles", conn).set_index("player_id")
    checked = tied_lowest = tied_best = 0
    for group, players in season.groupby("position_group"):
        n, label = len(players), fit.GROUP_LABELS[group]
        for metric in fit.ALLOWED_METRICS:
            column = fit.SEASON_COLUMNS[metric]
            values = players[column].to_numpy()
            for player_id, value in zip(players["player_id"], values):
                below = int((values < value).sum())
                level = int((values == value).sum()) - 1
                percentile = pctl.at[player_id, f"{column}_pctl"]
                derived = fit.players_ranked_below(percentile, n)
                phrase = fit.rank_phrase(percentile, n, label)
                if derived != below:
                    fail(f"{group} {metric} {player_id}: derived {derived}, counted {below}")
                if phrase != expected_phrase(below, n, label):
                    fail(f"{group} {metric} {player_id}: phrase {phrase!r}")
                tied_lowest += below == 0 and level > 0
                tied_best += below + level == n - 1 and level > 0
                checked += 1
    print(f"  {checked} player-metric pairs checked across {season['position_group'].nunique()} groups")
    print(f"  {tied_lowest} pairs are 'lowest' while tied with others at the bottom (wording covers ties)")
    print(f"  {tied_best} pairs are tied at the top, so none of them is called 'best'")
    for bad in (0.3, 1.5, -0.1):
        try:
            fit.players_ranked_below(bad, 23)
            fail(f"percentile {bad} for n=23 was accepted")
        except ValueError:
            pass


def check_fit_labels() -> None:
    print("2. Fit labels")
    cases = {
        100: "Excellent fit", 85: "Excellent fit", 84.96: "Excellent fit", 84.94: "Strong fit",
        84.9: "Strong fit", 70: "Strong fit", 69.94: "Moderate fit", 69.9: "Moderate fit",
        55: "Moderate fit", 54.9: "Limited fit", 0: "Limited fit",
    }
    for score, label in cases.items():
        if fit.fit_label(score) != label:
            fail(f"fit_label({score}) = {fit.fit_label(score)!r}, expected {label!r}")
    print(f"  {len(cases)} boundary scores checked")


def check_fact_sheets(conn: sqlite3.Connection) -> None:
    print("3. Fact sheets for every player in the reference pool")
    season = pd.read_sql_query("SELECT * FROM player_season_p90", conn).set_index("player_id")
    pctl = pd.read_sql_query("SELECT * FROM player_percentiles", conn).set_index("player_id")
    reliability = metric_reliability(conn, 600)
    tiers = dict(zip(reliability["metric"], reliability["tier"]))
    sheets = 0
    caveat_counts = {"minutes": 0, "reliability": 0, "pool": 0, "transfer": 0}
    for group, weights in REQUIREMENTS.items():
        result = fit.fit_scores(conn, {"position_group": group, "weights": weights}, top_n=500)
        table = result.table
        total = sum(weights.values())
        for row in table.itertuples(index=False):
            sheet = build_fact_sheet(conn, result, row.player_id)
            s, p = season.loc[row.player_id], pctl.loc[row.player_id]
            sheets += 1
            where = f"{row.player}"

            if sheet["player"]["minutes"] != str(int(round(s["minutes"]))):
                fail(f"{where}: minutes {sheet['player']['minutes']}")
            if sheet["shortlist"]["rank"] != str(row.rank) or sheet["fit"]["score"] != f"{row.fit_score:.1f}":
                fail(f"{where}: rank or score")
            if sheet["shortlist"]["reference_pool"] != str(len(table)):
                fail(f"{where}: pool size")

            def check_entry(entry: dict) -> None:
                metric = entry["metric"]
                column = fit.SEASON_COLUMNS[metric]
                raw = s[column]
                value = f"{raw:.1f}%" if metric == "pass_completion_pct" else f"{raw:.2f}"
                if entry["value"] != value:
                    fail(f"{where} {metric}: value {entry['value']} != {value}")
                if entry["percentile"] != str(int(round(p[f"{column}_pctl"] * 100))):
                    fail(f"{where} {metric}: percentile {entry['percentile']}")
                if entry["reliability"] != tiers[metric]:
                    fail(f"{where} {metric}: tier")

            for entry in sheet["weighted_metrics"]:
                check_entry(entry)
                share = f"{weights[entry['metric']] / total * 100:.0f}%"
                if entry["weight"] != share:
                    fail(f"{where} {entry['metric']}: weight {entry['weight']} != {share}")
            pct = {m: p[f"{fit.SEASON_COLUMNS[m]}_pctl"] for m in fit.ALLOWED_METRICS}
            top = sorted(pct.values(), reverse=True)[:3]
            bottom = sorted(pct.values())[:3]
            got_top = [pct[e["metric"]] for e in sheet["strengths"]]
            got_bottom = [pct[e["metric"]] for e in sheet["weaknesses"]]
            if got_top != top or got_bottom != bottom:
                fail(f"{where}: strengths or weaknesses are not the 3 highest / lowest")
            for entry in sheet["strengths"] + sheet["weaknesses"]:
                check_entry(entry)

            caveats = " ".join(sheet["caveats"])
            low_share = sum(weights[m] for m in weights if tiers[m] == "low") / total
            expected = {
                "minutes": s["minutes"] < 900,
                "reliability": low_share > 0.5,
                "pool": len(table) < 30,
                "transfer": "," in s["teams"],
            }
            present = {
                "minutes": "limited minutes" in caveats,
                "reliability": "low reliability" in caveats,
                "pool": "Small reference pool" in caveats,
                "transfer": "mid-season move" in caveats,
            }
            for name in expected:
                if expected[name] != present[name]:
                    fail(f"{where}: caveat {name} expected {expected[name]}")
                caveat_counts[name] += present[name]
            if len(sheet["caveats"]) != sum(expected.values()):
                fail(f"{where}: {len(sheet['caveats'])} caveats")
    print(f"  {sheets} fact sheets checked; caveats raised: {caveat_counts}")


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        check_rank_phrases(conn)
        check_fit_labels()
        check_fact_sheets(conn)
    conn.close()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed.")
        sys.exit(1)
    print("All report checks passed.")


if __name__ == "__main__":
    main()
