"""Scoring and summaries for the parser and report evaluation.

Everything here works on plain records (dicts), so it can be tested without
the API. scripts/run_evaluation.py produces the records by running the real
parser and report generator; this module scores each run, summarises the
runs and writes the plain-language summary.

PacedCaller wraps the model call used during an evaluation: it pauses
before every call, counts calls and failed calls, and turns a rate-limit
error into RateLimitStop, which the parser and report generator do not
catch, so the run stops instead of quietly falling back to offline rules.
"""

import math
import statistics
import time
from collections import Counter

from src.llm import LLMError

# Thresholds for the PASS/FAIL table.
GROUP_ACCURACY_MIN = 0.90
QUALITY_RECALL_MIN = 0.80
FIRST_ATTEMPT_OK_MIN = 0.90
UNAVAILABLE_MAX = 0.10
RATE_LIMIT_MARKER = "rate limit"
REJECTED = "rejected"


class RateLimitStop(Exception):
    """The provider's rate limit or quota was reached; the run should stop and resume later.

    Deliberately not an LLMError, so parse_requirement and generate_report
    let it through instead of falling back to the offline path.
    """


class PacedCaller:
    """Wrap a call_llm-style function with a pause before every call and call counts.

    calls counts every attempted call, unavailable counts calls that raised
    an LLMError other than a rate limit, and paused_seconds is the measured
    time spent waiting, so latencies can leave it out.
    """

    def __init__(self, call, pause_seconds: float, sleep=time.sleep):
        self._call = call
        self._sleep = sleep
        self.pause_seconds = pause_seconds
        self.calls = 0
        self.unavailable = 0
        self.paused_seconds = 0.0
        self.errors: list[str] = []

    def __call__(self, system: str, user: str, *args, **kwargs) -> str:
        if self.pause_seconds > 0:
            started = time.perf_counter()
            self._sleep(self.pause_seconds)
            self.paused_seconds += time.perf_counter() - started
        self.calls += 1
        try:
            return self._call(system, user, *args, **kwargs)
        except LLMError as error:
            if RATE_LIMIT_MARKER in str(error).lower():
                raise RateLimitStop(str(error)) from None
            self.unavailable += 1
            self.errors.append(str(error))
            raise


def top_metrics(weights: dict) -> list[str]:
    """Return the metric(s) with the highest weight, sorted, so ties compare equal."""
    if not weights:
        return []
    best = max(weights.values())
    return sorted(metric for metric, weight in weights.items() if math.isclose(weight, best))


def score_parser_run(case: dict, record: dict) -> dict:
    """Return the scores for one parser run against its case.

    record has outcome ("requirement" or "rejected"), group, weights and
    min_minutes. For a reject case only reject_ok is scored. For a valid
    case, a rejection counts as a wrong group and every quality as missed.
    """
    rejected = record["outcome"] == REJECTED
    if case["should_reject"]:
        return {"reject_ok": rejected}
    weights = {} if rejected else record["weights"]
    qualities = [any(metric in weights for metric in options) for options in case["required_qualities"]]
    expected_minutes = case.get("expected_min_minutes")
    return {
        "wrongly_rejected": rejected,
        "group_ok": (not rejected) and record["group"] in case["accepted_groups"],
        "qualities_met": qualities,
        "minutes_ok": None if expected_minutes is None else (not rejected and record["min_minutes"] == expected_minutes),
        "top_metrics": top_metrics(weights),
    }


def _share(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _latency(values: list[float]) -> dict:
    """Return the median and 90th percentile (nearest rank) of latencies, in seconds."""
    if not values:
        return {"median_s": None, "p90_s": None, "n": 0}
    ordered = sorted(values)
    p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]
    return {"median_s": statistics.median(ordered), "p90_s": p90, "n": len(ordered)}


def _answer(record: dict) -> str:
    """Return the parser's actual answer in a short readable form."""
    if record["outcome"] == REJECTED:
        return f"rejected: {record.get('rejection')}"
    weights = ", ".join(f"{m} {w:.2f}" for m, w in sorted(record["weights"].items(), key=lambda kv: -kv[1]))
    return f"{record['group']}; {weights}; min {record['min_minutes']}"


def summarize_parser(cases: list[dict], runs: list[dict]) -> dict:
    """Summarise parser runs. Accuracy figures use model-answered runs only."""
    by_id = {case["id"]: case for case in cases}
    model = [run for run in runs if run["source"] == "llm"]
    offline = [run for run in runs if run["source"] == "offline"]
    valid = [run for run in model if not by_id[run["case_id"]]["should_reject"]]
    rejects = [run for run in model if by_id[run["case_id"]]["should_reject"]]
    required = sum(len(run["scores"]["qualities_met"]) for run in valid)
    met = sum(sum(run["scores"]["qualities_met"]) for run in valid)
    minute_runs = [run for run in valid if run["scores"]["minutes_ok"] is not None]

    per_case: dict[str, list[dict]] = {}
    for run in valid:
        per_case.setdefault(run["case_id"], []).append(run)
    repeated = {cid: rs for cid, rs in per_case.items() if len(rs) >= 2}
    same_group = [
        cid for cid, rs in repeated.items() if len({REJECTED if r["outcome"] == REJECTED else r["group"] for r in rs}) == 1
    ]
    same_top = [cid for cid, rs in repeated.items() if len({tuple(r["scores"]["top_metrics"]) for r in rs}) == 1]

    misses = []
    for run in model:
        case, scores = by_id[run["case_id"]], run["scores"]
        reasons = []
        if case["should_reject"] and not scores["reject_ok"]:
            reasons.append("should have been rejected")
        if not case["should_reject"]:
            if scores["wrongly_rejected"]:
                reasons.append("valid request rejected")
            elif not scores["group_ok"]:
                reasons.append(f"group not in {case['accepted_groups']}")
            missed = [
                " or ".join(options)
                for options, ok in zip(case["required_qualities"], scores["qualities_met"])
                if not ok
            ]
            if missed and not scores["wrongly_rejected"]:
                reasons.append(f"missed quality: {'; '.join(missed)}")
            if scores["minutes_ok"] is False and not scores["wrongly_rejected"]:
                reasons.append(f"min_minutes should be {case['expected_min_minutes']}")
        if reasons:
            misses.append(
                {"case_id": run["case_id"], "run": run["run"], "text": case["text"],
                 "reasons": reasons, "answer": _answer(run)}
            )

    return {
        "runs": len(runs),
        "model_answered": len(model),
        "offline_fallbacks": len(offline),
        "fallback_rate": _share(len(offline), len(runs)),
        "fallback_reasons": dict(Counter(run.get("fallback_reason") or "unknown" for run in offline)),
        "group_accuracy": _share(sum(run["scores"]["group_ok"] for run in valid), len(valid)),
        "group_runs": len(valid),
        "quality_recall": _share(met, required),
        "qualities_met": met,
        "qualities_required": required,
        "minutes_correct": _share(sum(run["scores"]["minutes_ok"] for run in minute_runs), len(minute_runs)),
        "minutes_runs": len(minute_runs),
        "rejections_correct": _share(sum(run["scores"]["reject_ok"] for run in rejects), len(rejects)),
        "reject_runs": len(rejects),
        "wrongly_rejected": sum(run["scores"]["wrongly_rejected"] for run in valid),
        "consistent_group_cases": len(same_group),
        "consistent_top_metric_cases": len(same_top),
        "repeated_cases": len(repeated),
        "inconsistent_group": sorted(set(repeated) - set(same_group)),
        "inconsistent_top_metric": sorted(set(repeated) - set(same_top)),
        "latency": _latency([run["latency_s"] for run in model]),
        "misses": misses,
        "offline_runs": [
            {"case_id": run["case_id"], "run": run["run"], "reason": run.get("fallback_reason"), "answer": _answer(run)}
            for run in offline
        ],
    }


def summarize_reports(runs: list[dict]) -> dict:
    """Summarise report runs: first-attempt and final fact-check results, fallbacks, latency."""
    first = Counter(run["first_status"] or "no model draft" for run in runs)
    drafts = [run for run in runs if run["first_status"] is not None]
    first_ok = sum(run["first_status"] != "failed" for run in drafts)
    return {
        "reports": len(runs),
        "first_attempt": dict(first),
        "model_drafts": len(drafts),
        "first_attempt_without_failure": _share(first_ok, len(drafts)),
        "final": dict(Counter(run["final_status"] for run in runs)),
        "regenerated": sum(run["regenerated"] for run in runs),
        "offline": sum(run["source"] == "offline" for run in runs),
        "model_calls": sum(run["model_calls"] for run in runs),
        "latency": _latency([run["latency_s"] for run in runs]),
        "first_attempt_failures": [
            {"case_id": run["case_id"], "rank": run["rank"], "player": run["player"],
             "problems": run["first_problems"]}
            for run in runs if run["first_status"] == "failed"
        ],
        "offline_reports": [
            {"case_id": run["case_id"], "rank": run["rank"], "player": run["player"], "notes": run["notes"]}
            for run in runs if run["source"] == "offline"
        ],
        "warnings": dict(Counter(w for run in runs for w in run["final_warnings"])),
    }


def threshold_table(parser: dict | None, reports: dict | None, calls: int, unavailable: int) -> list[dict]:
    """Return the PASS/FAIL rows; a measure with no data is "n/a"."""

    def row(name, value, limit, higher_is_better):
        if value is None:
            verdict = "n/a"
        else:
            verdict = "PASS" if (value >= limit if higher_is_better else value <= limit) else "FAIL"
        target = f"{'≥' if higher_is_better else '≤'} {limit:.0%}"
        return {"measure": name, "value": value, "target": target, "result": verdict}

    return [
        row("Parser group accuracy", parser and parser["group_accuracy"], GROUP_ACCURACY_MIN, True),
        row("Parser quality recall", parser and parser["quality_recall"], QUALITY_RECALL_MIN, True),
        row("First-attempt reports without failure", reports and reports["first_attempt_without_failure"],
            FIRST_ATTEMPT_OK_MIN, True),
        row("Unavailable model calls", _share(unavailable, calls), UNAVAILABLE_MAX, False),
    ]


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _seconds(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f} s"


def render_summary(meta: dict, parser: dict | None, reports: dict | None, thresholds: list[dict]) -> str:
    """Return summary.md: plain-language results, the threshold table and every miss."""
    lines = [
        f"# Evaluation: {meta['provider']} / {meta['model']}",
        "",
        f"Run on {meta['date']}. {meta['calls']} model calls in total, of which "
        f"{meta['unavailable']} failed (the model could not be reached or gave no usable reply)."
        + (" The run stopped early at a rate limit; results are partial." if meta.get("stopped") else ""),
        "",
        "## Against the targets",
        "",
        "| Measure | Result | Target | |",
        "|---|---|---|---|",
    ]
    lines += [f"| {t['measure']} | {_pct(t['value'])} | {t['target']} | **{t['result']}** |" for t in thresholds]

    if parser:
        p = parser
        lines += [
            "",
            "## Requirement parser",
            "",
            f"{p['runs']} runs: {p['model_answered']} answered by the model and {p['offline_fallbacks']} by the "
            f"offline keyword rules after a fallback ({_pct(p['fallback_rate'])}). The figures below use the "
            "model-answered runs only; a valid request that was rejected counts as a wrong group and as "
            "missing every quality.",
            "",
            f"- **Group accuracy:** {_pct(p['group_accuracy'])} of {p['group_runs']} valid-request runs got an accepted position group.",
            f"- **Quality recall:** {p['qualities_met']} of {p['qualities_required']} required qualities were weighted ({_pct(p['quality_recall'])}).",
            (f"- **Minutes handling:** {_pct(p['minutes_correct'])} of {p['minutes_runs']} runs set min_minutes as expected."
             if p["minutes_runs"] else "- **Minutes handling:** no case with an expected min_minutes in this run."),
            (f"- **Rejections:** {_pct(p['rejections_correct'])} of {p['reject_runs']} runs that should be rejected were rejected."
             if p["reject_runs"] else "- **Rejections:** no case that should be rejected in this run."),
            f"- **Valid requests wrongly rejected:** {p['wrongly_rejected']} of {p['group_runs']} runs.",
            f"- **Consistency:** the same group in every run for {p['consistent_group_cases']} of {p['repeated_cases']} "
            f"cases, and the same top-weighted metric for {p['consistent_top_metric_cases']} of {p['repeated_cases']}."
            + (f" Group varied in: {', '.join(p['inconsistent_group'])}." if p["inconsistent_group"] else "")
            + (f" Top metric varied in: {', '.join(p['inconsistent_top_metric'])}." if p["inconsistent_top_metric"] else "")
            if p["repeated_cases"] else "- **Consistency:** needs at least two runs per case (use --runs 2 or more).",
            f"- **Latency** (model-answered, pauses excluded): median {_seconds(p['latency']['median_s'])}, "
            f"90th percentile {_seconds(p['latency']['p90_s'])}.",
        ]
        if p["fallback_reasons"]:
            lines.append("- **Fallback reasons:** " + "; ".join(f"{r} ({n})" for r, n in p["fallback_reasons"].items()))
    if reports:
        r = reports
        first = r["first_attempt"]
        lines += [
            "",
            "## Scouting reports",
            "",
            f"{r['reports']} reports (each case's top 3 players).",
            "",
            f"- **First attempt:** {first.get('verified', 0)} verified, {first.get('warnings', 0)} with warnings only, "
            f"{first.get('failed', 0)} failed the fact check"
            + (f", {first['no model draft']} with no model draft" if first.get("no model draft") else "")
            + f". {_pct(r['first_attempt_without_failure'])} of model drafts had no failure.",
            "- **Final:** " + ", ".join(f"{n} {status}" for status, n in sorted(r["final"].items())) + ".",
            f"- **Needed a regeneration:** {r['regenerated']}. **Fell back to the offline template:** {r['offline']}.",
            f"- **Model calls:** {r['model_calls']} for {r['reports']} reports.",
            f"- **Latency** (pauses excluded): median {_seconds(r['latency']['median_s'])}, "
            f"90th percentile {_seconds(r['latency']['p90_s'])}.",
        ]
        if r["warnings"]:
            lines.append("- **Warnings in final reports:** " + "; ".join(f"{w} ({n})" for w, n in r["warnings"].items()))

    if parser and (parser["misses"] or parser["offline_runs"]):
        lines += ["", "## Parser misses (model-answered runs)", ""]
        if not parser["misses"]:
            lines.append("None.")
        for miss in parser["misses"]:
            lines.append(
                f"- **{miss['case_id']}** (run {miss['run']}): \"{miss['text']}\" — {'; '.join(miss['reasons'])}. "
                f"Model's answer: {miss['answer']}."
            )
        if parser["offline_runs"]:
            lines += ["", "Offline fallbacks (not scored above):", ""]
            lines += [
                f"- **{o['case_id']}** (run {o['run']}): {o['reason']}. Offline answer: {o['answer']}."
                for o in parser["offline_runs"]
            ]
    if reports and (reports["first_attempt_failures"] or reports["offline_reports"]):
        lines += ["", "## Report fact-check failures (first attempt)", ""]
        if not reports["first_attempt_failures"]:
            lines.append("None.")
        for fail in reports["first_attempt_failures"]:
            lines.append(f"- **{fail['case_id']} rank {fail['rank']}** ({fail['player']}): " + " ".join(fail["problems"]))
        if reports["offline_reports"]:
            lines += ["", "Reports written by the offline template:", ""]
            lines += [
                f"- **{o['case_id']} rank {o['rank']}** ({o['player']}): {' '.join(o['notes'])}"
                for o in reports["offline_reports"]
            ]
    return "\n".join(lines) + "\n"
