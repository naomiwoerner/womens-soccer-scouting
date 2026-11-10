"""Check a scouting report's text against the fact sheet it was written from.

check_grounding returns "failed" when the report states something the fact
sheet does not support (a number, a rank phrase, a number on the wrong
metric, a missing caveat or weighted metric), "warnings" for softer problems
(ordinals, quoted percentiles, outside-knowledge terms, a low-reliability
metric not named in the caveats), and "verified" otherwise. Nothing here
calls a model or reads the database.
"""

import re
from dataclasses import dataclass

from src.fit import METRIC_NAMES

# How each metric may be named in a report: its display name plus common
# short forms. Matched case-insensitively on word boundaries.
METRIC_ALIASES = {
    "np_shots": ("non-penalty shots",),
    "npxg": ("non-penalty expected goals", "npxg"),
    "xa": ("expected assists", "xa"),
    "key_passes": ("key passes",),
    "passes_completed": ("completed passes",),
    "pass_completion_pct": ("pass completion",),
    "progressive_passes": ("progressive passes",),
    "passes_into_final_third": ("passes into the final third",),
    "progressive_carries": ("progressive carries",),
    "successful_dribbles": ("successful dribbles",),
    "pressures": ("pressures",),
    "tackles_interceptions": ("tackles won and interceptions", "tackles and interceptions"),
    "ball_recoveries": ("ball recoveries",),
    "aerial_duels_won": ("aerial duels won", "aerial duels"),
}
assert set(METRIC_ALIASES) == set(METRIC_NAMES), "every allowed metric needs aliases"
_ALIAS_PATTERNS = {
    metric: re.compile(r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b", re.IGNORECASE)
    for metric, aliases in METRIC_ALIASES.items()
}

NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?%?")
# Numbers that may appear without being in the fact sheet: the season the
# system prompt names. (The "90" of "per 90" comes from the unit fields.)
CONTEXT_TEXT = ("2023/24",)
ORDINAL = re.compile(r"\b\d+(?:st|nd|rd|th)\b", re.IGNORECASE)
PERCENTILE_NUMBER = re.compile(
    r"\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\s*(?:-\s*)?percentile|percentile\s+(?:of\s+|rank\s+(?:of\s+)?)?\d+",
    re.IGNORECASE,
)
RANK_PHRASE = re.compile(
    r"tied for lowest of \d+|lowest of \d+|best of \d+|better than \d+ of \d+", re.IGNORECASE
)
OUTSIDE_TERMS = (
    "age", "aged", "born", "birthday", "years old", "year-old", "nationality", "national team",
    "international caps", "caps", "capped", "height", "tall", "cm", "centimetres", "centimeters",
    "contract", "transfer fee", "market value", "salary", "wage", "wages", "injury", "injuries",
    "injured", "youth academy", "reputation",
)
OUTSIDE_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(term) for term in OUTSIDE_TERMS) + r")\b", re.IGNORECASE
)
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
# Fact-sheet fields a report may not quote, so they are not sources for numbers.
UNQUOTABLE_FIELDS = ("percentile",)


@dataclass(frozen=True)
class GroundingResult:
    """The outcome of a grounding check.

    status is "verified", "warnings" or "failed". problems are the failures
    and warnings the softer issues, each a sentence that can be shown to a
    user or fed back to the model. numbers_checked counts the numbers found
    in the report text.
    """

    status: str
    problems: list[str]
    warnings: list[str]
    numbers_checked: int


def _strings(value, skip: tuple[str, ...] = ()) -> list[str]:
    """Return every string inside a nested fact-sheet value, leaving out skipped keys."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for key, item in value.items() if key not in skip for s in _strings(item, skip)]
    if isinstance(value, list):
        return [s for item in value for s in _strings(item, skip)]
    return []


def _numbers(text: str) -> list[str]:
    for context in CONTEXT_TEXT:
        text = text.replace(context, " ")
    return NUMBER.findall(text)


def _units(sections: dict) -> list[tuple[str, str]]:
    """Return (section, text) for every bullet, and every sentence of every paragraph."""
    units = []
    for name, value in sections.items():
        items = value if isinstance(value, list) else SENTENCE_END.split(value)
        units.extend((name, item.strip()) for item in items if item.strip())
    return units


def _section_text(value) -> str:
    return " ".join(value) if isinstance(value, list) else value


def metrics_named(text: str) -> set[str]:
    """Return the metrics a piece of text names, by display name or short form."""
    return {metric for metric, pattern in _ALIAS_PATTERNS.items() if pattern.search(text)}


def _metric_entries(fact_sheet: dict) -> dict[str, dict]:
    """Return one merged entry per metric from the weighted, strength and weakness lists."""
    entries: dict[str, dict] = {}
    for key in ("weighted_metrics", "strengths", "weaknesses"):
        for entry in fact_sheet[key]:
            entries.setdefault(entry["metric"], {}).update(entry)
    return entries


def _short(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def check_grounding(report) -> GroundingResult:
    """Check a ScoutingReport's sections against its fact sheet.

    Failures: a number that is not in the fact sheet (percentile fields do
    not count) or the scout's request; in a sentence or bullet naming exactly
    one metric, a number or rank phrase that belongs to a different metric; a
    rank phrase that is not in the fact sheet; a fact-sheet caveat missing
    from the caveats section; a weighted metric not named in fit_for_profile.
    Warnings: an ordinal or a number next to "percentile"; an
    outside-knowledge term; a low-reliability metric mentioned in the report
    but not named in the caveats section.
    """
    sheet, sections = report.fact_sheet, report.sections
    problems: list[str] = []
    warnings: list[str] = []
    entries = _metric_entries(sheet)

    # 1. Numbers: every number must come from the fact sheet or the request.
    sources = _strings(sheet, UNQUOTABLE_FIELDS)
    if getattr(report, "request", None):
        sources.append(report.request)
    allowed = {number for text in sources for number in _numbers(text)}
    units = _units(sections)
    numbers_checked = 0
    unsupported: dict[str, str] = {}
    for section, text in units:
        for number in _numbers(text):
            numbers_checked += 1
            if number not in allowed:
                unsupported.setdefault(number, section)
    for number, section in unsupported.items():
        problems.append(f'The number "{number}" in {section} is not in the fact sheet.')

    # 3. Rank phrases must match a fact-sheet rank phrase exactly.
    phrases = sorted({entry["rank"] for entry in entries.values()}, key=len, reverse=True)
    for section, text in units:
        for match in RANK_PHRASE.finditer(text):
            rest = text[match.start():].lower()
            if not any(rest.startswith(phrase.lower()) for phrase in phrases):
                problems.append(
                    f'The rank phrase "{_short(text[match.start():match.end() + 25])}" in {section} '
                    "does not match any rank phrase in the fact sheet."
                )

    # 2. Attribution: in a unit naming exactly one metric, its numbers and rank
    # phrases must be that metric's.
    general = {
        number
        for key in ("player", "shortlist", "fit", "caveats")
        for text in _strings(sheet[key])
        for number in _numbers(text)
    }
    metric_numbers = {
        metric: {n for field in ("value", "unit", "weight", "rank") if field in entry for n in _numbers(entry[field])}
        for metric, entry in entries.items()
    }
    for section, text in units:
        named = metrics_named(text)
        if len(named) != 1:
            continue
        (metric,) = named
        own = metric_numbers.get(metric, set())
        others = set().union(*(nums for m, nums in metric_numbers.items() if m != metric)) - own - general
        wrong = [n for n in _numbers(text) if n in others]
        name = METRIC_NAMES[metric]
        if wrong:
            problems.append(
                f"In {section}, \"{_short(text)}\" gives {', '.join(wrong)} for {name}, "
                f"but {'that belongs' if len(wrong) == 1 else 'those belong'} to another metric."
            )
        own_rank = entries.get(metric, {}).get("rank", "").lower()
        for match in RANK_PHRASE.finditer(text):
            rest = text[match.start():].lower()
            if any(rest.startswith(p.lower()) for p in phrases) and not (own_rank and rest.startswith(own_rank)):
                problems.append(
                    f"In {section}, \"{_short(text)}\" gives {name} a rank phrase that belongs to another metric."
                )

    # 4. Every caveat appears verbatim in the caveats section.
    caveat_text = " ".join(_section_text(sections.get("caveats", "")).split())
    for caveat in sheet["caveats"]:
        if " ".join(caveat.split()) not in caveat_text:
            problems.append(f'The caveats section is missing the caveat: "{caveat}"')

    # 5. Every weighted metric is named in fit_for_profile.
    fit_named = metrics_named(_section_text(sections.get("fit_for_profile", "")))
    for entry in sheet["weighted_metrics"]:
        if entry["metric"] not in fit_named:
            problems.append(f"fit_for_profile does not name the weighted metric {entry['name']}.")

    # 6-8. Warnings.
    for section, text in units:
        for match in ORDINAL.finditer(text):
            warnings.append(f'{section} uses the ordinal "{match.group()}"; write ranks as numbers.')
        for match in PERCENTILE_NUMBER.finditer(text):
            warnings.append(f'{section} quotes a percentile ("{match.group()}"); use rank phrases instead.')
        for match in OUTSIDE_PATTERN.finditer(text):
            warnings.append(
                f'{section} mentions "{match.group()}", which the fact sheet does not cover.'
            )
    mentioned = metrics_named(" ".join(_section_text(v) for k, v in sections.items() if k != "caveats"))
    in_caveats = metrics_named(caveat_text)
    for metric in sorted(mentioned - in_caveats):
        if entries.get(metric, {}).get("reliability") == "low":
            warnings.append(
                f"{METRIC_NAMES[metric]} has low reliability and is mentioned, "
                "but the caveats section does not name it."
            )

    status = "failed" if problems else "warnings" if warnings else "verified"
    return GroundingResult(status, problems, warnings, numbers_checked)

