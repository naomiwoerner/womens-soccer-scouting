"""Check the evaluation scoring, summaries and harness with simulated results (no API).

Run from the project root with: python scripts/run_evaluation_check.py
"""

import json
import sqlite3
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.report as report_module  # noqa: E402
import src.requirement_parser as parser_module  # noqa: E402
from scripts import run_evaluation as harness  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src.fit import fit_scores  # noqa: E402
from src.llm import LLMError  # noqa: E402

failures: list[str] = []

CASES = [
    {"id": "A", "text": "striker who shoots", "accepted_groups": ["Striker"],
     "required_qualities": [["np_shots"], ["npxg", "xa"]], "should_reject": False},
    {"id": "B", "text": "6 or CB", "accepted_groups": ["Defensive/Central Midfield", "Center Back"],
     "required_qualities": [["ball_recoveries"]], "expected_min_minutes": 1200, "should_reject": False},
    {"id": "C", "text": "vague winger", "accepted_groups": ["Attacking Midfield/Winger"],
     "required_qualities": [], "should_reject": False},
    {"id": "X", "text": "goalkeeper", "accepted_groups": [], "required_qualities": [], "should_reject": True},
]
BY_ID = {case["id"]: case for case in CASES}


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'ok' if condition else 'FAIL'}] {label}{f': {detail}' if detail else ''}")
    if not condition:
        failures.append(label)


def parsed(case_id, run, group, weights, minutes=600, source="llm", latency=1.0, reason=None):
    record = {"case_id": case_id, "run": run, "outcome": "requirement", "group": group, "weights": weights,
              "min_minutes": minutes, "source": source, "fallback_reason": reason, "latency_s": latency,
              "rejection": None, "model_calls": 1, "unavailable_calls": 0}
    record["scores"] = ev.score_parser_run(BY_ID[case_id], record)
    return record


def rejected(case_id, run, source="llm", latency=1.0, reason=None):
    record = {"case_id": case_id, "run": run, "outcome": "rejected", "group": None, "weights": {},
              "min_minutes": None, "source": source, "fallback_reason": reason, "latency_s": latency,
              "rejection": "Goalkeepers are excluded.", "model_calls": 1, "unavailable_calls": 0}
    record["scores"] = ev.score_parser_run(BY_ID[case_id], record)
    return record


def check_scoring() -> None:
    print("1. Scoring one parser run")
    s = ev.score_parser_run(CASES[0], {"outcome": "requirement", "group": "Striker",
                                       "weights": {"np_shots": 0.5, "xa": 0.5}, "min_minutes": 600})
    check("group right, both qualities met (xa counts for 'npxg or xa')",
          s["group_ok"] and s["qualities_met"] == [True, True] and s["minutes_ok"] is None)
    s = ev.score_parser_run(CASES[0], {"outcome": "requirement", "group": "Center Back",
                                       "weights": {"np_shots": 1.0}, "min_minutes": 600})
    check("wrong group, one quality missed", not s["group_ok"] and s["qualities_met"] == [True, False])
    s = ev.score_parser_run(CASES[1], {"outcome": "requirement", "group": "Center Back",
                                       "weights": {"ball_recoveries": 1.0}, "min_minutes": 1200})
    check("second accepted group, minutes matched", s["group_ok"] and s["minutes_ok"] is True)
    s = ev.score_parser_run(CASES[1], {"outcome": "requirement", "group": "Center Back",
                                       "weights": {"ball_recoveries": 1.0}, "min_minutes": 900})
    check("minutes not matched", s["minutes_ok"] is False)
    s = ev.score_parser_run(CASES[0], {"outcome": "rejected"})
    check("valid case rejected: wrong group, every quality missed",
          s["wrongly_rejected"] and not s["group_ok"] and s["qualities_met"] == [False, False])
    check("reject case rejected", ev.score_parser_run(CASES[3], {"outcome": "rejected"}) == {"reject_ok": True})
    check("reject case accepted",
          ev.score_parser_run(CASES[3], {"outcome": "requirement", "group": "Striker",
                                         "weights": {"npxg": 1.0}, "min_minutes": 600}) == {"reject_ok": False})
    check("top metrics: ties compare equal",
          ev.top_metrics({"a": 0.4, "b": 0.4, "c": 0.2}) == ["a", "b"] and ev.top_metrics({}) == [])


def check_parser_summary() -> dict:
    print("2. Parser summary")
    runs = [
        parsed("A", 1, "Striker", {"np_shots": 0.6, "npxg": 0.4}, latency=1.0),
        parsed("A", 2, "Striker", {"np_shots": 0.6, "npxg": 0.4}, latency=2.0),
        parsed("A", 3, "Striker", {"npxg": 0.6, "np_shots": 0.4}, latency=3.0),   # top metric changes
        parsed("B", 1, "Defensive/Central Midfield", {"ball_recoveries": 1.0}, minutes=1200, latency=4.0),
        parsed("B", 2, "Center Back", {"ball_recoveries": 1.0}, minutes=900, latency=5.0),  # group varies, minutes wrong
        rejected("B", 3, latency=6.0),                                                       # wrongly rejected
        parsed("C", 1, "Attacking Midfield/Winger", {"xa": 1.0}, latency=7.0),
        parsed("C", 2, "Striker", {"np_shots": 1.0}, source="offline", reason="The model did not respond."),
        rejected("X", 1, latency=8.0),
        parsed("X", 2, "Striker", {"npxg": 1.0}, latency=9.0),                              # should be rejected
    ]
    p = ev.summarize_parser(CASES, runs)
    check("model-answered vs offline", (p["model_answered"], p["offline_fallbacks"]) == (9, 1))
    check("fallback rate and reason", p["fallback_rate"] == 0.1 and p["fallback_reasons"] == {"The model did not respond.": 1})
    # valid model runs: A1 A2 A3 B1 B2 B3 C1 = 7; group ok: all but B3 = 6
    check("group accuracy 6 of 7", p["group_runs"] == 7 and abs(p["group_accuracy"] - 6 / 7) < 1e-12)
    # required: A 2x3=6 (all met), B 1x3=3 (met in B1, B2) -> 8 of 9; C has none
    check("quality recall 8 of 9", (p["qualities_met"], p["qualities_required"]) == (8, 9))
    check("minutes 1 of 3 (B3 rejected counts as wrong)", (p["minutes_runs"], p["minutes_correct"]) == (3, 1 / 3))
    check("rejections 1 of 2", (p["reject_runs"], p["rejections_correct"]) == (2, 0.5))
    check("wrongly rejected 1", p["wrongly_rejected"] == 1)
    check("consistency: group same for A only; top metric same for none",
          (p["repeated_cases"], p["consistent_group_cases"], p["consistent_top_metric_cases"]) == (2, 1, 0)
          and p["inconsistent_group"] == ["B"] and p["inconsistent_top_metric"] == ["A", "B"])
    check("latency median and p90 (nearest rank)", p["latency"] == {"median_s": 5.0, "p90_s": 9.0, "n": 9})
    missed_ids = sorted((m["case_id"], m["run"]) for m in p["misses"])
    check("every miss listed with the actual answer",
          missed_ids == [("B", 2), ("B", 3), ("X", 2)]
          and any("Striker; npxg 1.00; min 600" in m["answer"] for m in p["misses"]), str(missed_ids))
    check("offline run listed separately", [o["case_id"] for o in p["offline_runs"]] == ["C"])
    return p


def report_run(case_id, rank, first, final, source="llm", regenerated=False, calls=1, latency=2.0, problems=()):
    return {"case_id": case_id, "rank": rank, "player": f"Player {case_id}{rank}", "source": source,
            "first_status": first, "first_problems": list(problems), "final_status": final,
            "final_warnings": [], "regenerated": regenerated, "notes": [], "model_calls": calls, "latency_s": latency}


def check_report_summary() -> dict:
    print("3. Report summary")
    runs = [
        report_run("R1", 1, "verified", "verified"),
        report_run("R1", 2, "warnings", "warnings"),
        report_run("R1", 3, "failed", "verified", regenerated=True, calls=2, problems=["The number \"9\"..."]),
        report_run("R2", 1, "failed", "verified", source="offline", calls=2, problems=["missing caveat"]),
        report_run("R2", 2, None, "verified", source="offline", calls=1),
    ]
    r = ev.summarize_reports(runs)
    check("first attempt counts", r["first_attempt"] == {"verified": 1, "warnings": 1, "failed": 2, "no model draft": 1})
    check("first attempt without failure: 2 of 4 drafts", r["model_drafts"] == 4 and r["first_attempt_without_failure"] == 0.5)
    check("regenerations, offline, calls", (r["regenerated"], r["offline"], r["model_calls"]) == (1, 2, 7))
    check("failures listed", [f["case_id"] + str(f["rank"]) for f in r["first_attempt_failures"]] == ["R13", "R21"])
    return r


def check_thresholds_and_markdown(p: dict, r: dict) -> None:
    print("4. Thresholds and summary.md")
    rows = ev.threshold_table(p, r, calls=20, unavailable=3)
    results = {row["measure"]: row["result"] for row in rows}
    check("group accuracy 86% < 90%: FAIL", results["Parser group accuracy"] == "FAIL")
    check("quality recall 89% >= 80%: PASS", results["Parser quality recall"] == "PASS")
    check("first-attempt 50% < 90%: FAIL", results["First-attempt reports without failure"] == "FAIL")
    check("unavailable 15% > 10%: FAIL", results["Unavailable model calls"] == "FAIL")
    rows = ev.threshold_table(None, None, calls=0, unavailable=0)
    check("no data gives n/a", all(row["result"] == "n/a" for row in rows))
    check("exact boundaries pass", ev.threshold_table(
        {"group_accuracy": 0.9, "quality_recall": 0.8}, {"first_attempt_without_failure": 0.9}, 10, 1
    ) and all(row["result"] == "PASS" for row in ev.threshold_table(
        {"group_accuracy": 0.9, "quality_recall": 0.8}, {"first_attempt_without_failure": 0.9}, 10, 1)))
    md = ev.render_summary({"provider": "gemini", "model": "m", "date": "2026-11-10", "calls": 20, "unavailable": 3,
                            "stopped": True}, p, r, ev.threshold_table(p, r, 20, 3))
    for needle in ("| Parser group accuracy | 86% | ≥ 90% | **FAIL** |", "Group accuracy:** 86% of 7",
                   "8 of 9 required qualities", "results are partial", "**B** (run 3)", "Model's answer: rejected",
                   "Offline fallbacks (not scored above)", "**R1 rank 3**", "2 failed the fact check",
                   "1 with no model draft"):
        check(f"summary.md contains {needle!r}", needle in md)
    json.dumps({"parser": p, "reports": r})  # must be JSON-serialisable


def check_paced_caller() -> None:
    print("5. Paced, counted model calls")
    sleeps = []
    replies = iter(["ok", LLMError("The model did not respond within 30 seconds."),
                    LLMError("The Gemini API rate limit or free-tier quota was reached. Try again shortly.")])

    def fake_call(system, user):
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return reply

    caller = ev.PacedCaller(fake_call, 5.0, sleep=sleeps.append)
    check("returns the reply", caller("s", "u") == "ok")
    try:
        caller("s", "u")
        check("unavailable call re-raises LLMError", False)
    except LLMError:
        check("unavailable call re-raises LLMError", True)
    try:
        caller("s", "u")
        check("rate limit becomes RateLimitStop", False)
    except ev.RateLimitStop:
        check("rate limit becomes RateLimitStop", not issubclass(ev.RateLimitStop, LLMError))
    check("pause before every call, all counted",
          sleeps == [5.0, 5.0, 5.0] and (caller.calls, caller.unavailable) == (3, 1) and caller.paused_seconds < 0.1,
          f"paused_seconds {caller.paused_seconds:.4f} (measured; the fake sleep returns at once)")
    real = ev.PacedCaller(lambda system, user: "ok", 0.2)
    real("s", "u")
    check("measured pause with a real sleep", 0.2 <= real.paused_seconds < 0.4, f"{real.paused_seconds:.3f} s")


def check_harness() -> None:
    print("6. Harness with simulated replies")
    cases = json.loads((PROJECT_ROOT / "evaluation" / "parser_cases.json").read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in cases}
    good = '{"position_group": "Striker", "weights": {"pressures": 5}, "min_minutes": 600, "notes": "ok"}'

    def run(case_id, replies):
        sleeps = []
        caller = ev.PacedCaller(mock.Mock(side_effect=replies), 5.0, sleep=sleeps.append)
        spy = harness.OfflineSpy(parser_module.offline_parse)
        with mock.patch.object(parser_module, "call_llm", caller), mock.patch.object(parser_module, "offline_parse", spy):
            return harness.run_parser_case(by_id[case_id], 1, caller, spy)

    record = run("P03", [good])
    check("model answer scored as llm", record["source"] == "llm" and record["scores"]["group_ok"]
          and record["model_calls"] == 1 and 0 <= record["latency_s"] < 1.0, f"latency {record['latency_s']} s (pause excluded)")
    record = run("X01", ['{"error": "Goalkeepers are excluded from this dataset."}'])
    check("model rejection scored as llm, reject_ok", record["source"] == "llm" and record["scores"]["reject_ok"])
    record = run("P03", [LLMError("The model did not respond within 30 seconds.")])
    check("LLMError: offline fallback with reason", record["source"] == "offline"
          and record["fallback_reason"] == "The model did not respond within 30 seconds."
          and record["unavailable_calls"] == 1)
    record = run("X01", [LLMError("The model returned an empty reply.")])
    check("offline rejection after LLMError keeps the reason", record["source"] == "offline"
          and record["outcome"] == "rejected" and record["fallback_reason"] == "The model returned an empty reply.")
    rate = LLMError("The Gemini API rate limit or free-tier quota was reached. Try again shortly.")
    try:
        run("P03", [rate])
        check("rate limit stops the parser instead of falling back", False)
    except ev.RateLimitStop:
        check("rate limit stops the parser instead of falling back", True)

    with sqlite3.connect(PROJECT_ROOT / "data" / "scouting.db") as conn:
        report_case = json.loads((PROJECT_ROOT / "evaluation" / "report_cases.json").read_text(encoding="utf-8"))[0]
        requirement = {key: report_case[key] for key in ("position_group", "weights", "min_minutes")}
        shortlist = fit_scores(conn, requirement, top_n=3)
        row = shortlist.table.iloc[0]
        sheet = report_module.build_fact_sheet(conn, shortlist, int(row.player_id))
        grounded = json.dumps(report_module.offline_report(sheet).sections)
        caller = ev.PacedCaller(mock.Mock(side_effect=[grounded]), 5.0, sleep=lambda s: None)
        with mock.patch.object(report_module, "call_llm", caller), \
                mock.patch.object(report_module, "active_model", return_value=("gemini", "test")):
            record = harness.run_report(conn, report_case, 1, shortlist, row, caller)
        check("report record", (record["source"], record["first_status"], record["final_status"], record["model_calls"])
              == ("llm", "verified", "verified", 1))
        caller = ev.PacedCaller(mock.Mock(side_effect=[rate]), 5.0, sleep=lambda s: None)
        try:
            with mock.patch.object(report_module, "call_llm", caller):
                harness.run_report(conn, report_case, 1, shortlist, row, caller)
            check("rate limit stops the report instead of falling back", False)
        except ev.RateLimitStop:
            check("rate limit stops the report instead of falling back", True)
    conn.close()


def main() -> None:
    check_scoring()
    p = check_parser_summary()
    r = check_report_summary()
    check_thresholds_and_markdown(p, r)
    check_paced_caller()
    check_harness()
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print("All evaluation checks passed.")


if __name__ == "__main__":
    main()
