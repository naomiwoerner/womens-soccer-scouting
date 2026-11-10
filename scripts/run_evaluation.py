"""Evaluate the requirement parser and the scouting reports against the fixed cases.

Runs evaluation/parser_cases.json (--runs times each) and the top 3 players
of every case in evaluation/report_cases.json (--report-runs times), with
the configured provider and model. Every model call is preceded by a pause
(--pause seconds) to stay under the free-tier rate limit, and every call is
counted. Each finished run is appended to a JSONL file at once, so a run
stopped by a rate limit (or Ctrl-C) resumes where it left off when the same
command is run again (pass --out to resume a folder from an earlier day).

Output: results/evaluation/<provider>_<model>_<date>/ with parser_runs.jsonl,
report_runs.jsonl, summary.json and summary.md.

Run from the project root, for example:
python scripts/run_evaluation.py --part all --runs 3
"""

import argparse
import datetime as dt
import json
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.llm as llm  # noqa: E402
import src.report as report_module  # noqa: E402
import src.requirement_parser as parser_module  # noqa: E402
from src.evaluation import (  # noqa: E402
    PacedCaller,
    RateLimitStop,
    render_summary,
    score_parser_run,
    summarize_parser,
    summarize_reports,
    threshold_table,
)
from src.fit import RequirementError, fit_scores  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
CASES_DIR = PROJECT_ROOT / "evaluation"
RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"
REPORT_PLAYERS = 3
# Calls per run: parser 1 (or 2 with the JSON retry); report 1 (up to 3 with
# the JSON retry and one fact-check regeneration).
PARSER_CALLS, REPORT_CALLS = (1, 2), (1, 3)
OFFLINE_REASON_PREFIX = "Used the offline keyword parser. Reason: "
REGENERATED_PREFIX = "The first draft failed the fact check"


def shown(path: Path) -> str:
    """Return a path relative to the project when it is inside it, else as given."""
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as lines:
        return [json.loads(line) for line in lines if line.strip()]


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record, ensure_ascii=False) + "\n")


class OfflineSpy:
    """Wrap offline_parse to record whether parse_requirement fell back to it."""

    def __init__(self, offline_parse):
        self._offline_parse = offline_parse
        self.used = False

    def __call__(self, text):
        self.used = True
        return self._offline_parse(text)


def run_parser_case(case: dict, run: int, caller: PacedCaller, spy: OfflineSpy) -> dict:
    """Run one parser case once and return its scored record."""
    calls, unavailable, paused = caller.calls, caller.unavailable, caller.paused_seconds
    errors_before = len(caller.errors)
    spy.used = False
    record = {"case_id": case["id"], "run": run, "text": case["text"], "should_reject": case["should_reject"]}
    start = time.perf_counter()
    try:
        result = parser_module.parse_requirement(case["text"])
        requirement = result.requirement
        record.update(
            outcome="requirement",
            group=requirement["position_group"],
            weights=requirement["weights"],
            min_minutes=requirement["min_minutes"],
            notes=result.notes,
            raw_response=result.raw_response,
            rejection=None,
        )
        reason = result.notes[0][len(OFFLINE_REASON_PREFIX):] if spy.used and result.notes else None
    except RequirementError as error:
        record.update(outcome="rejected", group=None, weights={}, min_minutes=None, notes=[],
                      raw_response=None, rejection=str(error))
        new_errors = caller.errors[errors_before:]
        reason = (new_errors[-1] if new_errors else "the model's reply could not be used") if spy.used else None
    elapsed = time.perf_counter() - start - (caller.paused_seconds - paused)
    record.update(
        source="offline" if spy.used else "llm",
        fallback_reason=reason,
        model_calls=caller.calls - calls,
        unavailable_calls=caller.unavailable - unavailable,
        latency_s=round(elapsed, 3),
    )
    record["scores"] = score_parser_run(case, record)
    return record


def run_report(conn, case: dict, run: int, shortlist, row, caller: PacedCaller) -> dict:
    """Generate and fact-check one report and return its record."""
    calls, unavailable, paused = caller.calls, caller.unavailable, caller.paused_seconds
    start = time.perf_counter()
    report = report_module.generate_report(conn, case["request_text"], shortlist, int(row["player_id"]))
    elapsed = time.perf_counter() - start - (caller.paused_seconds - paused)
    first, final = report.first_grounding, report.grounding
    return {
        "case_id": case["id"],
        "run": run,
        "rank": int(row["rank"]),
        "player": row["player"],
        "source": report.source,
        "provider": report.provider,
        "model": report.model,
        "first_status": first.status if first else None,
        "first_problems": first.problems if first else [],
        "first_warnings": first.warnings if first else [],
        "final_status": final.status,
        "final_problems": final.problems,
        "final_warnings": final.warnings,
        "numbers_checked": final.numbers_checked,
        "regenerated": any(note.startswith(REGENERATED_PREFIX) for note in report.notes),
        "notes": report.notes,
        "model_calls": caller.calls - calls,
        "unavailable_calls": caller.unavailable - unavailable,
        "latency_s": round(elapsed, 3),
        "sections": report.sections,
    }


def write_summary(out_dir: Path, meta: dict, parser_cases: list[dict]) -> None:
    parser_runs = load_jsonl(out_dir / "parser_runs.jsonl")
    report_runs = load_jsonl(out_dir / "report_runs.jsonl")
    all_runs = parser_runs + report_runs
    meta = {
        **meta,
        "calls": sum(r["model_calls"] for r in all_runs),
        "unavailable": sum(r["unavailable_calls"] for r in all_runs),
    }
    parser = summarize_parser(parser_cases, parser_runs) if parser_runs else None
    reports = summarize_reports(report_runs) if report_runs else None
    thresholds = threshold_table(parser, reports, meta["calls"], meta["unavailable"])
    summary = {"meta": meta, "parser": parser, "reports": reports, "thresholds": thresholds}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "summary.md").write_text(render_summary(meta, parser, reports, thresholds), encoding="utf-8")


def main() -> None:
    args = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args.add_argument("--part", choices=("parser", "reports", "all"), default="all")
    args.add_argument("--runs", type=int, default=3, help="runs per parser case (default 3)")
    args.add_argument("--report-runs", type=int, default=1, help="runs per report (default 1: 30 reports)")
    args.add_argument("--pause", type=float, default=5.0, help="seconds to wait before every model call")
    args.add_argument("--limit", type=int, help="use only the first N cases of each file")
    args.add_argument("--out", type=Path, help="results folder (to resume a run from an earlier day)")
    options = args.parse_args()

    provider, model = llm.active_model()
    if not llm.has_api_key():
        sys.exit(f"No {llm.KEY_NAMES[provider]} found; the evaluation needs the model.")
    parser_cases = json.loads((CASES_DIR / "parser_cases.json").read_text(encoding="utf-8"))
    report_cases = json.loads((CASES_DIR / "report_cases.json").read_text(encoding="utf-8"))
    if options.limit:
        parser_cases, report_cases = parser_cases[: options.limit], report_cases[: options.limit]
    do_parser, do_reports = options.part in ("parser", "all"), options.part in ("reports", "all")

    today = dt.date.today().isoformat()
    out_dir = options.out or RESULTS_DIR / f"{provider}_{model.replace('/', '-')}_{today}"
    out_dir.mkdir(parents=True, exist_ok=True)
    parser_path, report_path = out_dir / "parser_runs.jsonl", out_dir / "report_runs.jsonl"
    done_parser = {(r["case_id"], r["run"]) for r in load_jsonl(parser_path)}
    done_reports = {(r["case_id"], r["run"], r["rank"]) for r in load_jsonl(report_path)}

    parser_todo = [
        (run, case) for run in range(1, options.runs + 1) for case in parser_cases
        if do_parser and (case["id"], run) not in done_parser
    ]
    report_todo = [
        (run, case, rank) for run in range(1, options.report_runs + 1) for case in report_cases
        for rank in range(1, REPORT_PLAYERS + 1)
        if do_reports and (case["id"], run, rank) not in done_reports
    ]
    low = len(parser_todo) * PARSER_CALLS[0] + len(report_todo) * REPORT_CALLS[0]
    high = len(parser_todo) * PARSER_CALLS[1] + len(report_todo) * REPORT_CALLS[1]
    print(f"Provider: {provider}; model: {model}; results: {shown(out_dir)}")
    print(
        f"To do: {len(parser_todo)} parser runs and {len(report_todo)} reports "
        f"({len(done_parser)} parser runs and {len(done_reports)} reports already saved)."
    )
    print(
        f"Expected model calls: {low} to {high} (usually close to {low}); "
        f"at least {low * options.pause / 60:.0f} min of pauses alone at {options.pause:g} s per call."
    )

    caller = PacedCaller(llm.call_llm, options.pause)
    spy = OfflineSpy(parser_module.offline_parse)
    parser_module.call_llm = caller
    report_module.call_llm = caller
    parser_module.offline_parse = spy
    meta = {"provider": provider, "model": model, "date": today, "stopped": False}

    stopped = None
    try:
        for index, (run, case) in enumerate(parser_todo, 1):
            record = run_parser_case(case, run, caller, spy)
            append_jsonl(parser_path, record)
            answer = record["group"] if record["outcome"] != "rejected" else "rejected"
            print(f"  parser {index}/{len(parser_todo)} {case['id']} run {run}: {record['source']}, {answer}, "
                  f"{record['model_calls']} call(s), {record['latency_s']:.1f} s")
        if report_todo:
            with sqlite3.connect(DB_PATH) as conn:
                shortlists = {}
                for index, (run, case, rank) in enumerate(report_todo, 1):
                    if case["id"] not in shortlists:
                        requirement = {key: case[key] for key in ("position_group", "weights", "min_minutes")}
                        shortlists[case["id"]] = fit_scores(conn, requirement, top_n=REPORT_PLAYERS)
                    shortlist = shortlists[case["id"]]
                    row = shortlist.table.iloc[rank - 1]
                    record = run_report(conn, case, run, shortlist, row, caller)
                    append_jsonl(report_path, record)
                    print(f"  report {index}/{len(report_todo)} {case['id']} rank {rank} ({record['player']}): "
                          f"{record['source']}, first {record['first_status']}, final {record['final_status']}, "
                          f"{record['model_calls']} call(s), {record['latency_s']:.1f} s")
            conn.close()
    except RateLimitStop as error:
        stopped = f"Stopped at a rate limit: {error}"
    except KeyboardInterrupt:
        stopped = "Stopped by Ctrl-C."

    meta["stopped"] = bool(stopped)
    write_summary(out_dir, meta, json.loads((CASES_DIR / "parser_cases.json").read_text(encoding="utf-8")))
    print(f"\n{caller.calls} model calls this session ({caller.unavailable} unavailable).")
    print(f"Summary: {shown(out_dir / 'summary.md')}")
    if stopped:
        resume = " ".join(sys.argv[1:])
        if not options.out:
            resume += f" --out {shown(out_dir)}"
        print(f"{stopped}\nFinished runs are saved. To resume, run:\n  python scripts/run_evaluation.py {resume}")
        sys.exit(2)


if __name__ == "__main__":
    main()
