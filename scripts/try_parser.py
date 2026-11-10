"""Try the requirement parser on example requests and show each shortlist.

Uses the configured model (LLM_PROVIDER, see src/llm.py) when its API key is
set; otherwise runs the offline keyword parser only and says so. Run from the project root with:
python scripts/try_parser.py
"""

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.fit import RequirementError, fit_scores  # noqa: E402
from src.llm import KEY_NAMES, LLMError, active_model, has_api_key  # noqa: E402
from src.requirement_parser import offline_parse, parse_requirement  # noqa: E402

DB_PATH = PROJECT_ROOT / "data" / "scouting.db"
TOP_N = 3

EXAMPLES = (
    "I need a ball-playing centre-back who progresses the ball and is strong in the air, "
    "with at least 900 minutes.",
    "A pressing number 9 who works hard off the ball and still gets on the end of chances.",
    "Find me a creative winger who can beat players 1v1.",
    "We need a new goalkeeper who is comfortable with the ball at her feet.",
    "What will the weather be like in Munich this weekend?",
)


def show(connection: sqlite3.Connection, text: str, use_llm: bool) -> None:
    """Parse one request and print the requirement and the top of its shortlist."""
    print(f"\n> {text}")
    try:
        result = parse_requirement(text) if use_llm else offline_parse(text)
    except RequirementError as error:
        print(f"  Rejected: {error}")
        return

    requirement = result.requirement
    weights = ", ".join(f"{name} {weight:.2f}" for name, weight in requirement["weights"].items())
    answered_by = f" ({result.provider}, {result.model})" if result.provider else ""
    print(f"  Source: {result.source}{answered_by}")
    print(f"  Group: {requirement['position_group']}; min_minutes: {requirement['min_minutes']}")
    print(f"  Weights: {weights}")
    for note in result.notes:
        print(f"  Note: {note}")

    shortlist = fit_scores(connection, requirement, top_n=TOP_N)
    print(
        f"  Top {TOP_N} of {shortlist.eligible_players} eligible "
        f"(reference pool {shortlist.pool_size}):"
    )
    for row in shortlist.table.itertuples(index=False):
        print(f"    {row.rank}. {row.player} ({row.team}), {row.minutes:.0f} min, fit {row.fit_score:.1f}")


def main() -> None:
    try:
        provider, model = active_model()
    except LLMError as error:
        print(f"Settings problem: {error}\nRunning the offline keyword parser only.")
        provider = model = None
    use_llm = provider is not None and has_api_key()
    if provider is not None:
        print(f"Provider: {provider}; model: {model}")
    if use_llm:
        print("API key found: parsing with the model, falling back to offline rules if needed.")
    elif provider is not None:
        print(f"No {KEY_NAMES[provider]} found: running the offline keyword parser only.")
    with sqlite3.connect(DB_PATH) as connection:
        for text in EXAMPLES:
            show(connection, text, use_llm)
    connection.close()


if __name__ == "__main__":
    main()
