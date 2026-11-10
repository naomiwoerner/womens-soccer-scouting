"""Print scouting reports for the top two players of a pressing-number-9 shortlist.

For each player, prints the fact sheet, the model's report (when an API key
is configured) and the offline template report. Run from the project root
with: python scripts/try_report.py
"""

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fit import fit_scores  # noqa: E402
from src.llm import KEY_NAMES, LLMError, active_model, has_api_key  # noqa: E402
from src.report import SECTIONS, build_fact_sheet, generate_report, offline_report  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
REQUEST_TEXT = "A pressing number 9 who works hard off the ball and gets plenty of shots away"
REQUIREMENT = {
    "position_group": "Striker",
    "weights": {"pressures": 5, "npxg": 4, "np_shots": 4, "ball_recoveries": 3},
    "min_minutes": 600,
}
RANKS = (1, 2)
RULE = "=" * 78


def print_report(title: str, report) -> None:
    print(f"\n--- {title} ---")
    for note in report.notes:
        print(f"Note: {note}")
    for name in SECTIONS:
        value = report.sections[name]
        print(f"\n{name.replace('_', ' ').capitalize()}")
        if isinstance(value, list):
            for item in value:
                print(f"  - {item}")
        else:
            print(f"  {value}")
    print_grounding("Fact check", report.grounding)
    if report.first_grounding is not None and report.first_grounding is not report.grounding:
        print_grounding("First draft's fact check", report.first_grounding)


def print_grounding(title: str, grounding) -> None:
    if grounding is None:
        return
    print(f"\n{title}: {grounding.status} ({grounding.numbers_checked} numbers checked)")
    for problem in grounding.problems:
        print(f"  problem: {problem}")
    for warning in grounding.warnings:
        print(f"  warning: {warning}")


def main() -> None:
    try:
        provider, model = active_model()
        use_llm = has_api_key()
        print(f"Provider: {provider}; model: {model}")
        if not use_llm:
            print(f"No {KEY_NAMES[provider]} found: printing offline reports only.")
    except LLMError as error:
        use_llm = False
        print(f"Settings problem: {error}\nPrinting offline reports only.")
    print(f"Request: {REQUEST_TEXT}")

    with sqlite3.connect(DB_PATH) as conn:
        shortlist = fit_scores(conn, REQUIREMENT, top_n=max(RANKS))
        for row in shortlist.table.itertuples(index=False):
            print(f"\n{RULE}\nRank {row.rank}: {row.player} ({row.team}), fit {row.fit_score:.1f}\n{RULE}")
            fact_sheet = build_fact_sheet(conn, shortlist, row.player_id)
            print("\n--- Fact sheet ---")
            print(json.dumps(fact_sheet, indent=2, ensure_ascii=False))
            if use_llm:
                report = generate_report(conn, REQUEST_TEXT, shortlist, row.player_id)
                answered_by = f"{report.provider}, {report.model}" if report.provider else "fallback"
                print_report(f"Model report ({report.source}; {answered_by})", report)
            print_report("Offline report", offline_report(fact_sheet))
    conn.close()


if __name__ == "__main__":
    main()
