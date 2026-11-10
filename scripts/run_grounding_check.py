"""Check src/grounding.py: no false alarms, and every rule catches its error.

1. Every offline report (one per player in the reference pool, under a
   requirement for her group) must come out "verified".
2. Simulated bad reports, each made by changing one thing in Fenna Kalma's
   offline report for a pressing number 9, must be caught by the right rule.

Run from the project root with: python scripts/run_grounding_check.py
"""

import copy
import sqlite3
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import fit  # noqa: E402
from src.grounding import check_grounding  # noqa: E402
from src.report import build_fact_sheet, offline_report  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
REQUEST_TEXT = "A pressing number 9 who works hard off the ball and gets plenty of shots away"
STRIKER = {"pressures": 5, "npxg": 4, "np_shots": 4, "ball_recoveries": 3}
REQUIREMENTS = {
    "Striker": STRIKER,
    "Center Back": {"aerial_duels_won": 2, "progressive_passes": 1, "pass_completion_pct": 1},
    "Full Back/Wing Back": {"progressive_carries": 1, "xa": 1, "tackles_interceptions": 1},
    "Defensive/Central Midfield": {"passes_completed": 1, "ball_recoveries": 1},
    "Attacking Midfield/Winger": {"xa": 1, "key_passes": 1, "successful_dribbles": 1},
}

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {label}{f': {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


def check_offline_reports(conn: sqlite3.Connection) -> None:
    print("1. Offline reports (no false alarms)")
    statuses, numbers = Counter(), 0
    for group, weights in REQUIREMENTS.items():
        result = fit.fit_scores(conn, {"position_group": group, "weights": weights}, top_n=500)
        for player_id, player in zip(result.table["player_id"], result.table["player"]):
            grounding = check_grounding(offline_report(build_fact_sheet(conn, result, int(player_id))))
            statuses[grounding.status] += 1
            numbers += grounding.numbers_checked
            if grounding.status != "verified":
                print(f"    {player}: {grounding.problems + grounding.warnings}")
    total = sum(statuses.values())
    check(
        f"all {total} offline reports are verified",
        statuses == Counter({"verified": total}),
        f"{dict(statuses)}, {numbers} numbers checked",
    )


def kalma_report(conn: sqlite3.Connection):
    result = fit.fit_scores(conn, {"position_group": "Striker", "weights": STRIKER}, top_n=500)
    player_id = int(result.table.loc[result.table["player"] == "Fenna Kalma", "player_id"].iloc[0])
    return offline_report(build_fact_sheet(conn, result, player_id))


def edited(report, **changes):
    """Return a copy of the report with some sections replaced."""
    sections = copy.deepcopy(report.sections)
    sections.update(changes)
    return replace(report, sections=sections)


def replaced(report, section: str, old: str, new: str):
    """Return a copy with old replaced by new in one section (paragraph or bullets)."""
    value = report.sections[section]
    if isinstance(value, list):
        assert any(old in item for item in value), (section, old)
        value = [item.replace(old, new) for item in value]
    else:
        assert old in value, (section, old)
        value = value.replace(old, new)
    return edited(report, **{section: value})


def expect(label: str, report, status: str, needle: str, in_warnings: bool = False) -> None:
    """Check the grounding status and that some problem (or warning) contains needle."""
    grounding = check_grounding(report)
    messages = grounding.warnings if in_warnings else grounding.problems
    hit = [m for m in messages if needle in m]
    check(label, grounding.status == status and bool(hit), f"{grounding.status}: {hit[0] if hit else messages}")


def check_bad_reports(conn: sqlite3.Connection) -> None:
    print("2. Simulated bad reports, one per rule")
    good = kalma_report(conn)
    check("Kalma's offline report is verified", check_grounding(good).status == "verified")
    strengths = good.sections["strengths"]
    shots, aerial = strengths[0], strengths[2]

    expect(
        "1 numbers: a wrong value",
        replaced(good, "strengths", "4.33 per 90", "4.38 per 90"),
        "failed",
        '"4.38"',
    )
    swapped = [shots.replace("4.33", "3.28"), strengths[1], aerial.replace("3.28", "4.33")]
    expect(
        "2 attribution: two metrics' values swapped",
        edited(good, strengths=swapped),
        "failed",
        "belongs to another metric",
    )
    fit_text = good.sections["fit_for_profile"]
    expect(
        "2 attribution: one metric's weight on another",
        edited(good, fit_for_profile=fit_text + " Pressures carry 25% of the score."),
        "failed",
        "gives 25% for pressures",
    )
    expect(
        "2 attribution: another metric's rank phrase",
        edited(good, playing_style="Her pressures are best of 23 strikers."),
        "failed",
        "rank phrase that belongs to another metric",
    )
    expect(
        "3 rank phrases: an invented phrase",
        replaced(good, "strengths", "better than 20 of 22 other strikers", "better than 19 of 22 other strikers"),
        "failed",
        "does not match any rank phrase",
    )
    expect(
        "2+3: a real rank phrase on the wrong metric",
        replaced(good, "strengths", "better than 20 of 22 other strikers", "better than 21 of 22 other strikers"),
        "failed",
        "rank phrase that belongs to another metric",
    )
    expect(
        "3 rank phrases: 'other' dropped",
        replaced(good, "strengths", "better than 20 of 22 other strikers", "better than 20 of 22 strikers"),
        "failed",
        "does not match any rank phrase",
    )
    caveats = good.sections["caveats"]
    expect(
        "4 caveats: one left out",
        edited(good, caveats=caveats[1:]),
        "failed",
        "missing the caveat",
    )
    expect(
        "5 weighted metrics: one left out of fit_for_profile",
        edited(good, fit_for_profile="Her pressures, npxG and non-penalty shots fit the request."),
        "failed",
        "does not name the weighted metric ball recoveries",
    )
    expect(
        "6 warnings: a quoted percentile",
        edited(good, playing_style=good.sections["playing_style"] + " Her shooting is in the 95th percentile."),
        "failed",
        "quotes a percentile",
        in_warnings=True,
    )
    expect(
        "6 warnings: an ordinal (the number itself is in the fact sheet)",
        replaced(good, "summary", "She ranks 1 of 23", "She ranks 1st of 23"),
        "warnings",
        'ordinal "1st"',
        in_warnings=True,
    )
    expect(
        "7 warnings: an invented age",
        edited(good, playing_style=good.sections["playing_style"] + " At her age she can still improve."),
        "warnings",
        '"age"',
        in_warnings=True,
    )
    expect(
        "7 warnings + 1 numbers: an invented age with a number",
        edited(good, summary=good.sections["summary"] + " She is 25 years old."),
        "failed",
        '"25"',
    )
    expect(
        "8 warnings: a low-reliability metric not named in the caveats",
        edited(good, caveats=caveats[:3]),
        "warnings",
        "successful dribbles has low reliability",
        in_warnings=True,
    )

    print("   Sources the check allows")
    with_request = replace(
        edited(good, summary=good.sections["summary"] + " She suits the request for a pressing number 9."),
        request=REQUEST_TEXT,
    )
    check("a number from the scout's request is allowed", check_grounding(with_request).status == "verified")
    check(
        "the same number without the request is not",
        check_grounding(replace(with_request, request=None)).status == "failed",
    )
    season = edited(good, summary=good.sections["summary"] + " This covers the 2023/24 season.")
    check("the season named in the system prompt is allowed", check_grounding(season).status == "verified")
    percentile_number = good.fact_sheet["weighted_metrics"][0]["percentile"]
    quoted = edited(good, summary=good.sections["summary"] + f" Pressing: {percentile_number}.")
    check(
        f"a percentile field ({percentile_number}) is not a source",
        check_grounding(quoted).status == "failed",
    )


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        check_offline_reports(conn)
        check_bad_reports(conn)
    conn.close()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All grounding checks passed.")


if __name__ == "__main__":
    main()
