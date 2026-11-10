"""Turn a scout's free-text request into a validated fit-score requirement.

parse_requirement asks the configured language model (see src.llm) first and
falls back to deterministic keyword rules (offline_parse) when the model is
unavailable or its reply cannot be used.
Every requirement, from either path, passes src.fit.validate_requirement.
"""

import json
import re
from dataclasses import dataclass, replace

from src.fit import (
    ALLOWED_METRICS,
    DEFAULT_MIN_MINUTES,
    MAX_MIN_MINUTES,
    MAX_WEIGHTED_METRICS,
    OUTFIELD_GROUPS,
    REQUIREMENT_KEYS,
    RequirementError,
    validate_requirement,
)
from src.llm import LLMError, active_model, call_llm, strip_code_fence
from src.metrics import METRICS

LLM_MIN_WEIGHT, LLM_MAX_WEIGHT = 1, 5
LLM_FIELDS = (*REQUIREMENT_KEYS, "notes")
OFFLINE_WEIGHT = 3
MAX_TEXT_CHARS = 2000
GOALKEEPER_MESSAGE = (
    "Goalkeepers are excluded from this dataset, so the tool can only search for outfield players."
)
MINUTES_PATTERN = re.compile(r"(\d[\d,]*)\s*\+?\s*(?:min|mins|minutes)\b")


@dataclass(frozen=True)
class ParseResult:
    """A parsed requirement and where it came from.

    requirement is validated, with weights normalized to sum to 1. source is
    "llm" or "offline". notes explain the interpretation and any fallback.
    raw_response is the model's last reply. provider and model name the model
    that answered, such as "gemini" and "gemini-3.8-flash". All three are None
    when the offline parser produced the result, including after a fallback.
    """

    requirement: dict
    source: str
    notes: list[str]
    raw_response: str | None
    provider: str | None = None
    model: str | None = None


class _InvalidReply(Exception):
    """The model's reply could not be used. The message is shown to the model on the retry."""


def build_system_prompt() -> str:
    """Return the system prompt, built from the groups and metric definitions in code."""
    definitions = {metric.name: metric.definition for metric in METRICS}
    groups = "\n".join(f"- {group}" for group in OUTFIELD_GROUPS)
    metrics = "\n".join(f"- {name}: {definitions[name]}" for name in ALLOWED_METRICS)
    return f"""You turn a football scout's description of the player they want into a requirement for a scouting tool. The tool covers outfield players in the 2023/24 Frauen-Bundesliga and ranks them by percentiles within their position group.

Position groups (choose exactly one, spelled exactly as shown):
{groups}

Metrics (use these names exactly; all are per 90 minutes except pass_completion_pct):
{metrics}

Reply with one JSON object and nothing else:
{{"position_group": "<group>", "weights": {{"<metric>": <weight>}}, "min_minutes": <minutes>, "notes": "<one or two sentences>"}}

Rules:
- weights: 1 to {MAX_WEIGHTED_METRICS} metrics that best capture what the scout asked for. Each weight is a whole number from {LLM_MIN_WEIGHT} (minor) to {LLM_MAX_WEIGHT} (most important).
- position_group: the closest group for the role described. For example, a number 9 is a Striker, a number 6 is Defensive/Central Midfield and a winger is Attacking Midfield/Winger.
- min_minutes: {DEFAULT_MIN_MINUTES} unless the scout asks for more playing time. It must be a whole number from {DEFAULT_MIN_MINUTES} to {MAX_MIN_MINUTES}; if the scout asks for a value outside that range, use the nearest limit and say so in notes.
- notes: say briefly how you interpreted the request, including anything the metrics cannot measure, such as age, height or contract status.
- If the request is not about finding an outfield player (it asks for a goalkeeper, is not about scouting, or cannot be understood), reply only with {{"error": "<one sentence for the scout>"}}. For a goalkeeper, explain that goalkeepers are excluded from this dataset."""


SYSTEM_PROMPT = build_system_prompt()



def _read_reply(raw: str) -> tuple[dict, list[str]]:
    """Return the validated requirement and notes from one of the model's replies.

    Raises RequirementError when the model reports the request as out of scope,
    and _InvalidReply when the reply cannot be used as it stands.
    """
    try:
        data = json.loads(strip_code_fence(raw))
    except json.JSONDecodeError as error:
        raise _InvalidReply(f"The reply was not valid JSON ({error.msg}).") from None
    if not isinstance(data, dict):
        raise _InvalidReply("The reply must be a single JSON object.")
    if "error" in data:
        reason = data["error"]
        if isinstance(reason, str) and reason.strip():
            raise RequirementError(reason.strip())
        raise _InvalidReply('"error" must be a non-empty sentence.')

    unknown = [key for key in data if key not in LLM_FIELDS]
    if unknown:
        raise _InvalidReply(f"Unexpected field(s): {', '.join(unknown)}. Allowed: {', '.join(LLM_FIELDS)}.")
    weights = data.get("weights")
    if isinstance(weights, dict):
        out_of_range = [
            name
            for name, value in weights.items()
            if isinstance(value, bool)
            or not isinstance(value, int)
            or not LLM_MIN_WEIGHT <= value <= LLM_MAX_WEIGHT
        ]
        if out_of_range:
            raise _InvalidReply(
                f"Weights must be whole numbers from {LLM_MIN_WEIGHT} to {LLM_MAX_WEIGHT}; "
                f"check: {', '.join(out_of_range)}."
            )
    notes = data.get("notes")
    if notes is not None and not isinstance(notes, str):
        raise _InvalidReply('"notes" must be a string.')
    try:
        requirement = validate_requirement({key: data[key] for key in REQUIREMENT_KEYS if key in data})
    except RequirementError as error:
        raise _InvalidReply(str(error)) from None
    return requirement, [notes.strip()] if notes and notes.strip() else []


def _parse_with_llm(text: str) -> ParseResult:
    """Ask the model for a requirement, retrying once with the problem fed back."""
    provider, model = active_model()
    raw = call_llm(SYSTEM_PROMPT, text)
    try:
        requirement, notes = _read_reply(raw)
    except _InvalidReply as problem:
        retry = (
            f"Scout's request:\n{text}\n\nYour previous reply:\n{raw}\n\n"
            f"It could not be used: {problem}\nReply again with corrected JSON only."
        )
        raw = call_llm(SYSTEM_PROMPT, retry)
        requirement, notes = _read_reply(raw)
    return ParseResult(requirement, "llm", notes, raw, provider, model)


def _clean_text(text) -> str:
    """Return the request text, or raise RequirementError if it is empty or too long."""
    if not isinstance(text, str) or not text.strip():
        raise RequirementError("Describe the player you are looking for.")
    if len(text) > MAX_TEXT_CHARS:
        raise RequirementError(f"Keep the description under {MAX_TEXT_CHARS} characters.")
    return text.strip()


def parse_requirement(text: str) -> ParseResult:
    """Return a validated requirement for a free-text request.

    The model is asked first. If its reply is not valid JSON or fails
    validation, it gets one retry with the problem explained. If that also
    fails, or the model cannot be reached, the offline keyword parser is used and the notes say why.
    Raises RequirementError when the request is out of scope, such as a
    goalkeeper or a question unrelated to scouting.
    """
    text = _clean_text(text)
    try:
        return _parse_with_llm(text)
    except LLMError as error:
        reason = str(error)
    except _InvalidReply as error:
        reason = f"The model's reply could not be used after one retry ({error})"
    result = offline_parse(text)
    return replace(result, notes=[f"Used the offline keyword parser. Reason: {reason}", *result.notes])


GOALKEEPER_SYNONYMS = (
    "goalkeeper",
    "goal keeper",
    "keeper",
    "goalie",
    "gk",
    "shot stopper",
    "sweeper keeper",
)

GROUP_SYNONYMS = {
    "center back": "Center Back",
    "centre back": "Center Back",
    "center half": "Center Back",
    "centre half": "Center Back",
    "central defender": "Center Back",
    "cb": "Center Back",
    "stopper": "Center Back",
    "full back": "Full Back/Wing Back",
    "fullback": "Full Back/Wing Back",
    "wing back": "Full Back/Wing Back",
    "wingback": "Full Back/Wing Back",
    "right back": "Full Back/Wing Back",
    "left back": "Full Back/Wing Back",
    "rb": "Full Back/Wing Back",
    "lb": "Full Back/Wing Back",
    "rwb": "Full Back/Wing Back",
    "lwb": "Full Back/Wing Back",
    "defensive midfielder": "Defensive/Central Midfield",
    "holding midfielder": "Defensive/Central Midfield",
    "central midfielder": "Defensive/Central Midfield",
    "midfielder": "Defensive/Central Midfield",
    "number 6": "Defensive/Central Midfield",
    "number six": "Defensive/Central Midfield",
    "no. 6": "Defensive/Central Midfield",
    "#6": "Defensive/Central Midfield",
    "number 8": "Defensive/Central Midfield",
    "number eight": "Defensive/Central Midfield",
    "no. 8": "Defensive/Central Midfield",
    "#8": "Defensive/Central Midfield",
    "cdm": "Defensive/Central Midfield",
    "box to box": "Defensive/Central Midfield",
    "anchor": "Defensive/Central Midfield",
    "pivot": "Defensive/Central Midfield",
    "attacking midfielder": "Attacking Midfield/Winger",
    "winger": "Attacking Midfield/Winger",
    "wide forward": "Attacking Midfield/Winger",
    "inside forward": "Attacking Midfield/Winger",
    "wide player": "Attacking Midfield/Winger",
    "number 10": "Attacking Midfield/Winger",
    "number ten": "Attacking Midfield/Winger",
    "no. 10": "Attacking Midfield/Winger",
    "#10": "Attacking Midfield/Winger",
    "playmaker": "Attacking Midfield/Winger",
    "cam": "Attacking Midfield/Winger",
    "rw": "Attacking Midfield/Winger",
    "lw": "Attacking Midfield/Winger",
    "striker": "Striker",
    "center forward": "Striker",
    "centre forward": "Striker",
    "number 9": "Striker",
    "number nine": "Striker",
    "no. 9": "Striker",
    "#9": "Striker",
    "cf": "Striker",
    "forward": "Striker",
    "target woman": "Striker",
    "poacher": "Striker",
}

METRIC_SYNONYMS = {
    "press": ("pressures",),
    "pressing": ("pressures",),
    "high press": ("pressures",),
    "counter press": ("pressures",),
    "pressure": ("pressures",),
    "work rate": ("pressures",),
    "aerial": ("aerial_duels_won",),
    "header": ("aerial_duels_won",),
    "heading": ("aerial_duels_won",),
    "in the air": ("aerial_duels_won",),
    "creative": ("xa", "key_passes"),
    "creativity": ("xa", "key_passes"),
    "chance creation": ("xa", "key_passes"),
    "creates chances": ("xa", "key_passes"),
    "assist": ("xa",),
    "xa": ("xa",),
    "key pass": ("key_passes",),
    "goal": ("npxg",),
    "goal threat": ("npxg",),
    "scoring": ("npxg",),
    "scorer": ("npxg",),
    "finishing": ("npxg",),
    "finisher": ("npxg",),
    "chances": ("npxg",),
    "xg": ("npxg",),
    "npxg": ("npxg",),
    "shot": ("np_shots",),
    "shooting": ("np_shots",),
    "dribble": ("successful_dribbles",),
    "dribbling": ("successful_dribbles",),
    "dribbler": ("successful_dribbles",),
    "1v1": ("successful_dribbles",),
    "beat players": ("successful_dribbles",),
    "take on": ("successful_dribbles",),
    "carry": ("progressive_carries",),
    "carries": ("progressive_carries",),
    "carrying": ("progressive_carries",),
    "ball carrier": ("progressive_carries",),
    "progressive carries": ("progressive_carries",),
    "progressive": ("progressive_passes",),
    "progresses the ball": ("progressive_passes",),
    "progress the ball": ("progressive_passes",),
    "progressive passes": ("progressive_passes",),
    "line breaking": ("progressive_passes",),
    "ball playing": ("progressive_passes", "pass_completion_pct"),
    "build up": ("progressive_passes",),
    "distribution": ("progressive_passes",),
    "final third": ("passes_into_final_third",),
    "pass accuracy": ("pass_completion_pct",),
    "passing accuracy": ("pass_completion_pct",),
    "pass completion": ("pass_completion_pct",),
    "secure in possession": ("pass_completion_pct",),
    "retention": ("pass_completion_pct",),
    "passing volume": ("passes_completed",),
    "on the ball": ("passes_completed",),
    "tackle": ("tackles_interceptions",),
    "tackling": ("tackles_interceptions",),
    "interception": ("tackles_interceptions",),
    "ball winner": ("tackles_interceptions",),
    "ball winning": ("tackles_interceptions",),
    "wins the ball": ("tackles_interceptions",),
    "recovery": ("ball_recoveries",),
    "recoveries": ("ball_recoveries",),
    "second balls": ("ball_recoveries",),
}

DEFAULT_PROFILES = {
    "Center Back": ("progressive_passes", "pass_completion_pct", "aerial_duels_won", "tackles_interceptions"),
    "Full Back/Wing Back": (
        "progressive_passes",
        "passes_into_final_third",
        "progressive_carries",
        "tackles_interceptions",
    ),
    "Defensive/Central Midfield": (
        "passes_completed",
        "progressive_passes",
        "tackles_interceptions",
        "ball_recoveries",
    ),
    "Attacking Midfield/Winger": ("xa", "key_passes", "successful_dribbles", "progressive_carries"),
    "Striker": ("npxg", "np_shots", "xa", "pressures"),
}

_unknown = {m for metrics in METRIC_SYNONYMS.values() for m in metrics} | {
    m for metrics in DEFAULT_PROFILES.values() for m in metrics
}
assert _unknown <= set(ALLOWED_METRICS), f"Synonyms name metrics that are not allowed: {_unknown - set(ALLOWED_METRICS)}"
assert set(GROUP_SYNONYMS.values()) == set(OUTFIELD_GROUPS) == set(DEFAULT_PROFILES)


def _normalize(text: str) -> str:
    """Return lowercase text with hyphens as spaces, so 'Centre-Back' reads 'centre back'."""
    return text.lower().replace("-", " ")


def _phrase_matches(text: str, phrases) -> list[tuple[int, str]]:
    """Return (position, phrase) for each phrase found, longest phrases first, never overlapping.

    Phrases match whole words, with an optional plural "s" or "es".
    """
    taken = [False] * len(text)
    found = []
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?:s|es)?(?![a-z0-9])"
        for match in re.finditer(pattern, text):
            if any(taken[match.start() : match.end()]):
                continue
            taken[match.start() : match.end()] = [True] * (match.end() - match.start())
            found.append((match.start(), phrase))
    return sorted(found)


def offline_parse(text: str) -> ParseResult:
    """Return a requirement from deterministic keyword rules, without calling a model.

    The first position mentioned sets the group. Each recognised quality maps
    to one or more metrics with equal weight, up to six in order of mention; if
    none are recognised, a general profile for the group is used. A number
    followed by "minutes" sets min_minutes, moved to the nearest allowed limit
    if it falls outside 600-1200, as the system prompt instructs. The result
    passes validate_requirement.
    """
    text = _clean_text(text)
    normalized = _normalize(text)
    if _phrase_matches(normalized, GOALKEEPER_SYNONYMS):
        raise RequirementError(GOALKEEPER_MESSAGE)

    notes = []
    groups = list(dict.fromkeys(GROUP_SYNONYMS[p] for _, p in _phrase_matches(normalized, GROUP_SYNONYMS)))
    if not groups:
        raise RequirementError(
            "Could not find a position in the request. Mention one, for example 'centre back', "
            "'full back', 'number 6', 'winger' or 'striker'."
        )
    if len(groups) > 1:
        notes.append(f"Several positions were mentioned ({', '.join(groups)}); used the first, {groups[0]}.")

    metrics = list(
        dict.fromkeys(m for _, p in _phrase_matches(normalized, METRIC_SYNONYMS) for m in METRIC_SYNONYMS[p])
    )
    if not metrics:
        metrics = list(DEFAULT_PROFILES[groups[0]])
        notes.append(f"No specific qualities were recognised, so a general {groups[0]} profile was used.")
    if len(metrics) > MAX_WEIGHTED_METRICS:
        notes.append(f"Only the first {MAX_WEIGHTED_METRICS} qualities mentioned were kept.")
        metrics = metrics[:MAX_WEIGHTED_METRICS]
    notes.append("Offline keyword rules give every recognised quality the same weight.")

    request = {"position_group": groups[0], "weights": {m: OFFLINE_WEIGHT for m in metrics}}
    minutes = MINUTES_PATTERN.search(normalized)
    if minutes:
        asked = int(minutes.group(1).replace(",", ""))
        used = min(max(asked, DEFAULT_MIN_MINUTES), MAX_MIN_MINUTES)
        if used != asked:
            notes.append(
                f"{asked} minutes is outside the allowed {DEFAULT_MIN_MINUTES}-{MAX_MIN_MINUTES}; "
                f"used {used}."
            )
        request["min_minutes"] = used
    return ParseResult(validate_requirement(request), "offline", notes, None)
