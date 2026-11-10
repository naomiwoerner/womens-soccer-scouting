"""Check the report's display helpers and fact sheets against the database.

1. rank_phrase / players_ranked_below: for every player and every allowed
   metric, the count derived from PERCENT_RANK must equal the number of
   players in her group with a strictly lower raw value, counted here in
   Python from player_season_p90.
2. fit_label at and around each cutoff.
3. build_fact_sheet for every player in the reference pool, under a
   requirement for her group: formatted numbers, shortlist rank, strengths,
   weaknesses and caveats, each recomputed independently.
4. offline_report for every fact sheet: six well-formed sections, every
   caveat quoted, no percentiles.
5. generate_report with simulated model replies: valid, fenced, one retry
   with the problem fed back, offline fallback, and the request description
   built from the weights when no text is given.

Run from the project root with: python scripts/run_report_check.py
"""

import json
import re
import sqlite3
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src import fit  # noqa: E402
from src.eda import metric_reliability  # noqa: E402
from src import report  # noqa: E402
from src.llm import LLMError  # noqa: E402
from src.report import build_fact_sheet, offline_report  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
REQUEST_TEXT = "A pressing number 9 who works hard off the ball and gets plenty of shots away"
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


def expected_phrase(below: int, level: int, n: int, label: str) -> str:
    """The wording rule, written out again independently of src/fit.py.

    below is how many players have a strictly lower raw value; level is how
    many other players share her value.
    """
    if below == n - 1:
        return f"best of {n} {label}"
    if below == 0:
        return f"{'tied for lowest' if level > 0 else 'lowest'} of {n} {label}"
    return f"better than {below} of {n - 1} other {label}"


def raw_counts(values, value) -> tuple[int, int]:
    """Return (players strictly below, other players level) for one raw value."""
    return int((values < value).sum()), int((values == value).sum()) - 1


def check_rank_phrases(conn: sqlite3.Connection) -> None:
    print("1. Rank phrases against raw values")
    season = pd.read_sql_query("SELECT * FROM player_season_p90", conn)
    pctl = pd.read_sql_query("SELECT * FROM player_percentiles", conn).set_index("player_id")
    checked = outright_lowest = tied_lowest = tied_best = 0
    for group, players in season.groupby("position_group"):
        n, label = len(players), fit.GROUP_LABELS[group]
        for metric in fit.ALLOWED_METRICS:
            column = fit.SEASON_COLUMNS[metric]
            values = players[column].to_numpy()
            for player_id, value in zip(players["player_id"], values):
                below, level = raw_counts(values, value)
                percentile = pctl.at[player_id, f"{column}_pctl"]
                derived = fit.players_ranked_below(percentile, n)
                phrase = fit.rank_phrase(percentile, n, label, tied_for_lowest=below == 0 and level > 0)
                if derived != below:
                    fail(f"{group} {metric} {player_id}: derived {derived}, counted {below}")
                if phrase != expected_phrase(below, level, n, label):
                    fail(f"{group} {metric} {player_id}: phrase {phrase!r}")
                outright_lowest += below == 0 and level == 0
                tied_lowest += below == 0 and level > 0
                tied_best += below + level == n - 1 and level > 0
                checked += 1
    print(f"  {checked} player-metric pairs checked across {season['position_group'].nunique()} groups")
    print(f"  {outright_lowest} pairs are outright 'lowest', {tied_lowest} are 'tied for lowest'")
    print(f"  {tied_best} pairs are tied at the top, so none of them is called 'best'")
    if not outright_lowest or not tied_lowest:
        fail("the data should contain both outright and tied lowest values")
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


def check_fact_sheets(conn: sqlite3.Connection) -> list[dict]:
    print("3. Fact sheets for every player in the reference pool")
    season = pd.read_sql_query("SELECT * FROM player_season_p90", conn).set_index("player_id")
    pctl = pd.read_sql_query("SELECT * FROM player_percentiles", conn).set_index("player_id")
    reliability = metric_reliability(conn, 600)
    tiers = dict(zip(reliability["metric"], reliability["tier"]))
    sheets, all_sheets = 0, []
    caveat_counts = {"minutes": 0, "reliability": 0, "pool": 0, "transfer": 0}
    entry_counts = {"tied": 0}
    for group, weights in REQUIREMENTS.items():
        result = fit.fit_scores(conn, {"position_group": group, "weights": weights}, top_n=500)
        table = result.table
        total = sum(weights.values())
        group_rows = season[season["position_group"] == group]
        n, label = len(group_rows), fit.GROUP_LABELS[group]
        for row in table.itertuples(index=False):
            sheet = build_fact_sheet(conn, result, row.player_id)
            all_sheets.append(sheet)
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
                below, level = raw_counts(group_rows[column].to_numpy(), raw)
                if entry["rank"] != expected_phrase(below, level, n, label):
                    fail(f"{where} {metric}: rank {entry['rank']!r}")
                entry_counts["tied"] += entry["rank"].startswith("tied for lowest")

            for entry in sheet["weighted_metrics"]:
                check_entry(entry)
                share = f"{weights[entry['metric']] / total * 100:.0f}%"
                if entry["weight"] != share:
                    fail(f"{where} {entry['metric']}: weight {entry['weight']} != {share}")
                column = fit.SEASON_COLUMNS[entry["metric"]]
                below, _ = raw_counts(group_rows[column].to_numpy(), s[column])
                match = (
                    "clear match" if 4 * below >= 3 * (n - 1)
                    else "partial match" if 2 * below >= n - 1
                    else "no match"
                )
                if entry["match"] != match:
                    fail(f"{where} {entry['metric']}: match {entry['match']!r} != {match!r}")
                entry_counts[match] = entry_counts.get(match, 0) + 1
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
    print(f"  {entry_counts['tied']} fact-sheet entries say 'tied for lowest', each confirmed from raw values")
    matches = {k: v for k, v in entry_counts.items() if k != 'tied'}
    print(f"  weighted-metric match labels confirmed from raw counts: {matches}")
    if not entry_counts["tied"]:
        fail("no fact sheet exercised the 'tied for lowest' wording")
    return all_sheets


ORDINAL = re.compile(r"\b\d+(?:st|nd|rd|th)\b")
RANK_COUNT = re.compile(r"^better than (\d+) of (\d+) other ")


def share_from_phrase(phrase: str) -> float:
    """Return the share of other players she ranks above, read back from a rank phrase."""
    if phrase.startswith("best of "):
        return 1.0
    if phrase.startswith(("lowest of ", "tied for lowest of ")):
        return 0.0
    below, others = RANK_COUNT.match(phrase).groups()
    return int(below) / int(others)


def join_names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def check_offline_reports(sheets: list[dict]) -> None:
    print("4. Offline reports for every fact sheet")
    verdict_counts = {"clear": 0, "partly": 0, "not": 0}
    for sheet in sheets:
        name = sheet["player"]["name"]
        sections = offline_report(sheet).sections
        if tuple(sections) != report.SECTIONS:
            fail(f"{name}: offline sections {tuple(sections)}")
        for section, value in sections.items():
            if isinstance(value, list):
                ok = report.MIN_BULLETS <= len(value) <= report.MAX_BULLETS and all(v.strip() for v in value)
            else:
                ok = isinstance(value, str) and bool(value.strip())
            if not ok:
                fail(f"{name}: offline section {section} is malformed")
        text = json.dumps(sections, ensure_ascii=False)
        caveat_text = json.dumps(sections["caveats"], ensure_ascii=False)
        mentioned = sheet["weighted_metrics"] + sheet["strengths"] + sheet["weaknesses"]
        unflagged = {
            e["name"] for e in mentioned if e["reliability"] == "low" and e["name"] not in " ".join(sheet["caveats"])
        }
        expected_caveats = list(sheet["caveats"]) or ([] if unflagged else [report.NO_CAVEATS])
        for caveat in expected_caveats:
            if json.dumps(caveat, ensure_ascii=False)[1:-1] not in caveat_text:
                fail(f"{name}: caveat missing from offline report")
        for metric_name in unflagged:
            if metric_name.lower() not in caveat_text.lower() or "low reliability at 600 minutes" not in caveat_text:
                fail(f"{name}: low-reliability {metric_name} not named in the caveats")
        if unflagged and report.NO_CAVEATS in caveat_text:
            fail(f"{name}: says no caveats while naming low-reliability metrics")
        # Caveats may mention percentiles as a concept; the template's own wording may not.
        own_text = json.dumps({k: v for k, v in sections.items() if k != "caveats"}, ensure_ascii=False)
        if "percentile" in own_text:
            fail(f"{name}: offline report mentions percentiles outside the caveats")
        for entry in sheet["weighted_metrics"]:
            if entry["weight"] not in text or entry["value"] not in text:
                fail(f"{name}: offline report lacks {entry['metric']} weight or value")
        if ORDINAL.search(text):
            fail(f"{name}: offline report uses an ordinal")
        fit_text = json.dumps(sections["fit_for_profile"], ensure_ascii=False)
        verdicts = {"clear": [], "partly": [], "not": []}
        for entry in sheet["weighted_metrics"]:
            share = share_from_phrase(entry["rank"])
            key = "clear" if share >= 0.75 else "not" if share < 0.5 else "partly"
            verdicts[key].append(entry["name"])
            verdict_counts[key] += 1
        clauses = {
            "clear": "clearly meets the request on {}, where she is better than at least three quarters",
            "partly": "only partly on {}",
            "not": "on {}, where she is in the lower half" if (verdicts["clear"] or verdicts["partly"])
            else "in the lower half of the other {group} on {}",
        }
        group = fit.GROUP_LABELS[sheet["player"]["position_group"]]
        for key, names in verdicts.items():
            if names and clauses[key].format(join_names(names), group=group) not in fit_text:
                fail(f"{name}: fit_for_profile does not say {key} for {names}")
        if not verdicts["clear"] and "does not" not in fit_text:
            fail(f"{name}: fit_for_profile should say no quality is clearly met")
    print(f"  {len(sheets)} offline reports: six well-formed sections, every caveat quoted, no percentiles")
    print(f"  closing fit sentence matches rank phrases; verdicts per weighted metric: {verdict_counts}")


VALID_REPORT = {
    "summary": "Summary.",
    "strengths": ["One.", "Two."],
    "weaknesses": ["One.", "Two."],
    "playing_style": "Her numbers suggest a pressing forward.",
    "fit_for_profile": ["One.", "Two."],
    "caveats": ["One.", "Two."],
}


def run_generator(conn, result, player_id, replies, request_text=REQUEST_TEXT):
    """Run generate_report with simulated model replies (strings or exceptions)."""
    with (
        mock.patch.object(report, "call_llm", side_effect=replies) as call,
        mock.patch.object(report, "active_model", return_value=("gemini", "gemini-test")),
    ):
        produced = report.generate_report(conn, request_text, result, player_id)
    return produced, call


def check_generator(conn: sqlite3.Connection) -> None:
    print("5. Report generator with simulated model replies")
    requirement = {"position_group": "Striker", "weights": REQUIREMENTS["Striker"], "min_minutes": 600}
    result = fit.fit_scores(conn, requirement, top_n=1)
    player_id = int(result.table["player_id"].iloc[0])
    valid = json.dumps(VALID_REPORT)
    bad_json, missing = "not json", json.dumps({k: v for k, v in VALID_REPORT.items() if k != "caveats"})
    empty_list = json.dumps({**VALID_REPORT, "strengths": []})
    extra_key = json.dumps({**VALID_REPORT, "rating": "A"})
    non_string = json.dumps({**VALID_REPORT, "weaknesses": ["One.", 2]})

    def expect(label, replies, source, calls, check_extra=None, request_text=REQUEST_TEXT):
        produced, call = run_generator(conn, result, player_id, replies, request_text)
        ok = produced.source == source and call.call_count == calls
        if source == "llm":
            ok = ok and (produced.provider, produced.model) == ("gemini", "gemini-test")
            ok = ok and produced.sections == VALID_REPORT
        else:
            ok = ok and (produced.provider, produced.model) == (None, None)
            ok = ok and produced.sections == offline_report(produced.fact_sheet).sections
        if check_extra is not None:
            ok = ok and check_extra(produced, call)
        print(f"  [{'ok' if ok else 'FAIL'}] {label}: {produced.source}, {call.call_count} call(s)")
        if not ok:
            fail(f"generator: {label}")
        return produced, call

    def sends_request_and_sheet(produced, call):
        system, message = call.call_args_list[0].args
        sheet = json.dumps(produced.fact_sheet, indent=2, ensure_ascii=False)
        return system == report.SYSTEM_PROMPT and REQUEST_TEXT in message and sheet in message

    def retry_explains(problem):
        def check(produced, call):
            retry = call.call_args_list[1].args[1]
            return "Your previous reply" in retry and problem in retry and REQUEST_TEXT in retry
        return check

    def fallback_note(reason):
        def check(produced, call):
            return produced.notes[0].startswith(f"Used the offline report template. Reason: {reason}")
        return check

    expect("valid reply", [valid], "llm", 1, sends_request_and_sheet)
    expect("fenced reply", [f"```json\n{valid}\n```"], "llm", 1)
    expect("invalid JSON, then valid", [bad_json, valid], "llm", 2, retry_explains("not valid JSON"))
    expect("missing section, then valid", [missing, valid], "llm", 2, retry_explains("Missing section(s): caveats"))
    expect("unexpected key, then valid", [extra_key, valid], "llm", 2, retry_explains("Unexpected key(s): rating"))
    expect("non-string bullet, then valid", [non_string, valid], "llm", 2, retry_explains('"weaknesses" must be'))
    expect(
        "empty section twice: offline",
        [empty_list, empty_list],
        "offline",
        2,
        fallback_note("The model's reply could not be used after one retry"),
    )
    expect(
        "model unavailable: offline",
        [LLMError("The model did not respond within 30 seconds.")],
        "offline",
        1,
        fallback_note("The model did not respond within 30 seconds."),
    )
    expect(
        "model unavailable on the retry: offline",
        [bad_json, LLMError("The model returned an empty reply.")],
        "offline",
        2,
        fallback_note("The model returned an empty reply."),
    )
    description = "Striker; weights: pressures 31%, npxG 25%, non-penalty shots 25%, ball recoveries 19%; at least 600 minutes"

    def uses_description(produced, call):
        message = call.call_args_list[0].args[1]
        return f"Scout's request:\n{description}\n" in message and description in produced.notes[0]

    expect("no request text: description from weights", [valid], "llm", 1, uses_description, request_text=None)
    expect("blank request text: description from weights", [valid], "llm", 1, uses_description, request_text="  ")
    prompt_ok = all(f'"{name}"' in report.SYSTEM_PROMPT for name in report.SECTIONS) and (
        'Do not quote the "percentile" fields' in report.SYSTEM_PROMPT
    ) and "Never use ordinals" in report.SYSTEM_PROMPT and "\"match\" field" in report.SYSTEM_PROMPT
    print(f"  [{'ok' if prompt_ok else 'FAIL'}] system prompt names all six sections, uses the match field, bans percentiles and ordinals")
    if not prompt_ok:
        fail("system prompt")


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        check_rank_phrases(conn)
        check_fit_labels()
        sheets = check_fact_sheets(conn)
        check_offline_reports(sheets)
        check_generator(conn)
    conn.close()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed.")
        sys.exit(1)
    print("All report checks passed.")


if __name__ == "__main__":
    main()
