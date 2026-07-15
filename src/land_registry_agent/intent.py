"""Hybrid natural-language request interpretation and policy validation."""

from __future__ import annotations

import re
from typing import Protocol

from land_registry_agent.models import (
    Intent,
    InterpretationMethod,
    NoteDetailLevel,
    PersistenceMode,
)


class IntentFallback(Protocol):
    """Interface implemented by the structured-output LLM service."""

    def interpret(self, request: str, partial_intent: Intent) -> Intent:
        """Complete or correct an ambiguous deterministic interpretation."""


POSTCODE_PATTERN = re.compile(
    r"\b("
    r"GIR\s?0AA|"
    r"[A-PR-UWYZ][A-HK-Y]?\d[A-Z\d]?"
    r"(?:\s*\d[ABD-HJLNP-UW-Z]{2})?"
    r")\b",
    re.IGNORECASE,
)

VALID_POSTCODE_PATTERN = re.compile(
    r"^(?:"
    r"GIR 0AA|"
    r"[A-PR-UWYZ][A-HK-Y]?\d[A-Z\d]?"
    r"(?: \d[ABD-HJLNP-UW-Z]{2})?"
    r")$",
)

YEAR_PATTERN = re.compile(
    r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)"
    r"\s*(?:years?|yrs?)\b",
    re.IGNORECASE,
)

WORD_NUMBERS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

REGION_ALIASES = {
    "north east": "North East",
    "north-east": "North East",
    "north west": "North West",
    "north-west": "North West",
    "yorkshire and the humber": "Yorkshire and The Humber",
    "east midlands": "East Midlands",
    "west midlands": "West Midlands",
    "east of england": "East of England",
    "london": "London",
    "south east": "South East",
    "south-east": "South East",
    "south west": "South West",
    "south-west": "South West",
    "england": "England",
    "wales": "Wales",
    "scotland": "Scotland",
    "northern ireland": "Northern Ireland",
    "united kingdom": "United Kingdom",
    "uk": "United Kingdom",
}

COMPARISON_PATTERN = re.compile(
    r"\b(?:compare|comparison|versus|vs\.?|regional average|regional trend)\b",
    re.IGNORECASE,
)

STREET_RANKING_PATTERN = re.compile(
    r"\b(?:"
    r"highest[- ]value streets?|"
    r"most expensive streets?|"
    r"top streets?|"
    r"rank(?:ed|ing)? streets?"
    r")\b",
    re.IGNORECASE,
)

NOTE_NEGATION_PATTERN = re.compile(
    r"\b(?:no research note|without (?:a )?note|do not draft|"
    r"don't draft|analysis only)\b",
    re.IGNORECASE,
)

PERSISTENCE_FORBIDDEN_PATTERN = re.compile(
    r"\b(?:"
    r"do not|don't|should not|nothing should be"
    r")\s+(?:be\s+)?(?:save|saved|write|written|persist|persisted|add|added)\b"
    r"|\bwithout persistence\b"
    r"|\bnothing (?:should be )?written yet\b",
    re.IGNORECASE,
)

PERSISTENCE_REQUESTED_PATTERN = re.compile(
    r"\b(?:"
    r"save|persist|write|"
    r"add (?:it|this|the report|the note)|"
    r"tracking sheet"
    r")\b",
    re.IGNORECASE,
)

LIVE_REFRESH_PATTERN = re.compile(
    r"\b(?:live refresh|refresh live|subscribe|subscription|continuous updates?)\b",
    re.IGNORECASE,
)

COMPARISON_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:against|benchmark|contrast|relative to|"
    r"wider (?:area|region)|stacks? up)\b",
    re.IGNORECASE,
)

STREET_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:priciest|costliest|street leaderboard|"
    r"valuable streets?|best streets?)\b",
    re.IGNORECASE,
)

PERSISTENCE_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:track(?:er)?|record|log|file|store|sheet)\b",
    re.IGNORECASE,
)

NOTE_OMISSION_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:skip|omit|exclude|leave out|without)\b"
    r".{0,30}\b(?:note|summary|paragraph|write[- ]?up|report)\b",
    re.IGNORECASE,
)

NOTE_PARAGRAPH_COUNT_PATTERN = re.compile(
    r"\b([1-5]|one|two|three|four|five)"
    r"\s*[- ]paragraphs?\b",
    re.IGNORECASE,
)

CONCISE_NOTE_PATTERN = re.compile(
    r"\b(?:brief|concise|short)\s+"
    r"(?:research\s+)?(?:note|summary|report|write[- ]?up)\b",
    re.IGNORECASE,
)

DETAILED_NOTE_PATTERN = re.compile(
    r"\b(?:detailed|comprehensive|in[- ]depth)\s+"
    r"(?:research\s+)?(?:note|summary|report|write[- ]?up)\b",
    re.IGNORECASE,
)

NOTE_STYLE_AMBIGUITY_PATTERN = re.compile(
    r"\b(?:deep dive|executive summary|full write[- ]?up|"
    r"extended summary|more detailed account)\b",
    re.IGNORECASE,
)

NEGATED_COMPARISON_PATTERN = re.compile(
    r"\b(?:no|not|without|skip|omit|don't|do not)\b"
    r".{0,30}\b(?:compare|comparison|regional)\b",
    re.IGNORECASE,
)

NEGATED_STREET_PATTERN = re.compile(
    r"\b(?:no|not|without|skip|omit|don't|do not)\b"
    r".{0,30}\b(?:rank|ranking|streets?)\b",
    re.IGNORECASE,
)


def parse_deterministically(request: str) -> Intent:
    """Extract obvious supported values without calling a language model."""

    if not request.strip():
        raise ValueError("Request must not be blank.")

    postcode_match = POSTCODE_PATTERN.search(request)
    postcode = postcode_match.group(1) if postcode_match else None

    year_match = YEAR_PATTERN.search(request)
    requested_years = (
        _parse_number(year_match.group(1), default=3)
        if year_match is not None
        else 3
    )

    region = _find_region(request)
    comparison_requested = bool(COMPARISON_PATTERN.search(request))
    street_ranking_requested = bool(STREET_RANKING_PATTERN.search(request))
    note_requested = not bool(NOTE_NEGATION_PATTERN.search(request))
    note_detail_level, note_paragraph_count = _parse_note_preferences(request)
    persistence_mode = _parse_persistence_mode(request)

    confidence = 0.95
    clarification_reason: str | None = None

    if postcode is None:
        confidence = 0.45
        clarification_reason = "A UK postcode or postcode district is required."
    elif comparison_requested and region is None:
        confidence = 0.65
        clarification_reason = (
            "A region is required for the requested regional comparison."
        )

    return Intent(
        postcode=postcode,
        requested_years=requested_years,
        region=region,
        regional_comparison_requested=comparison_requested,
        street_ranking_requested=street_ranking_requested,
        note_requested=note_requested,
        note_detail_level=note_detail_level,
        note_paragraph_count=(
            note_paragraph_count if note_requested else None
        ),
        persistence_mode=persistence_mode,
        latest_available_data=_requests_latest_data(request),
        live_refresh_requested=bool(LIVE_REFRESH_PATTERN.search(request)),
        interpretation_method=InterpretationMethod.DETERMINISTIC,
        confidence=confidence,
        clarification_reason=clarification_reason,
    )


def interpret_request(
    request: str,
    fallback: IntentFallback | None = None,
) -> Intent:
    """Interpret a request, using an injected LLM only when necessary."""

    deterministic_intent = parse_deterministically(request)

    if not needs_llm_fallback(request, deterministic_intent):
        return apply_intent_policy(deterministic_intent)

    if fallback is None:
        return apply_intent_policy(deterministic_intent)

    llm_intent = fallback.interpret(request, deterministic_intent)
    llm_intent = _reconcile_llm_intent(
        request,
        deterministic_intent,
        llm_intent,
    )
    llm_intent = llm_intent.model_copy(
        update={"interpretation_method": InterpretationMethod.LLM_ASSISTED}
    )
    return apply_intent_policy(llm_intent)


def needs_llm_fallback(request: str, intent: Intent) -> bool:
    """Detect unresolved required fields or ambiguous optional actions."""

    required_fields_unresolved = intent.postcode is None or (
        intent.regional_comparison_requested and intent.region is None
    )

    optional_action_unresolved = (
        (
            not intent.regional_comparison_requested
            and COMPARISON_AMBIGUITY_PATTERN.search(request) is not None
        )
        or (
            not intent.street_ranking_requested
            and STREET_AMBIGUITY_PATTERN.search(request) is not None
        )
        or (
            intent.persistence_mode is PersistenceMode.NOT_REQUESTED
            and PERSISTENCE_AMBIGUITY_PATTERN.search(request) is not None
        )
        or (
            intent.note_requested
            and NOTE_OMISSION_AMBIGUITY_PATTERN.search(request) is not None
        )
        or (
            intent.note_requested
            and NOTE_STYLE_AMBIGUITY_PATTERN.search(request) is not None
        )
    )

    deterministic_result_conflicted = (
        (
            intent.regional_comparison_requested
            and NEGATED_COMPARISON_PATTERN.search(request) is not None
        )
        or (
            intent.street_ranking_requested
            and NEGATED_STREET_PATTERN.search(request) is not None
        )
    )

    return (
        required_fields_unresolved
        or optional_action_unresolved
        or deterministic_result_conflicted
    )


def _reconcile_llm_intent(
    request: str,
    partial_intent: Intent,
    llm_intent: Intent,
) -> Intent:
    """Constrain LLM output using explicit deterministic evidence."""

    note_omission_requested = (
        NOTE_NEGATION_PATTERN.search(request) is not None
        or NOTE_OMISSION_AMBIGUITY_PATTERN.search(request) is not None
    )

    note_requested = llm_intent.note_requested
    if partial_intent.note_requested and not note_omission_requested:
        note_requested = True

    explicit_note_preferences = _has_explicit_note_preferences(request)
    note_detail_level = (
        partial_intent.note_detail_level
        if explicit_note_preferences
        else llm_intent.note_detail_level
    )
    note_paragraph_count = (
        partial_intent.note_paragraph_count
        if explicit_note_preferences
        else llm_intent.note_paragraph_count
    )
    if not note_requested:
        note_paragraph_count = None

    return llm_intent.model_copy(
        update={
            "note_requested": note_requested,
            "note_detail_level": note_detail_level,
            "note_paragraph_count": note_paragraph_count,
            "region": (
                llm_intent.region
                if llm_intent.regional_comparison_requested
                else None
            ),
        }
    )


def apply_intent_policy(intent: Intent) -> Intent:
    """Validate an interpreted intent against the bounded workflow policy."""

    issues: list[str] = []

    if intent.postcode is None:
        issues.append("A UK postcode or postcode district is required.")
    elif not VALID_POSTCODE_PATTERN.fullmatch(intent.postcode):
        issues.append("The postcode format is not recognised.")

    if intent.regional_comparison_requested and intent.region is None:
        issues.append("A region is required for regional comparison.")

    if (
        intent.persistence_mode is PersistenceMode.REQUESTED
        and not intent.note_requested
    ):
        issues.append("Persistence requires a research note.")

    if intent.live_refresh_requested:
        issues.append(
            "Live resource subscriptions are outside the demonstration workflow."
        )

    clarification_reason = " ".join(issues) if issues else None
    confidence = min(intent.confidence, 0.49) if issues else intent.confidence

    return intent.model_copy(
        update={
            "clarification_reason": clarification_reason,
            "confidence": confidence,
        }
    )


def _parse_number(value: str, *, default: int) -> int:
    """Convert a numeric or simple word-based number expression."""

    lowered = value.lower()
    return WORD_NUMBERS.get(
        lowered,
        int(value) if value.isdigit() else default,
    )


def _parse_note_preferences(
    request: str,
) -> tuple[NoteDetailLevel, int | None]:
    """Extract bounded note detail and paragraph-count preferences."""

    paragraph_match = NOTE_PARAGRAPH_COUNT_PATTERN.search(request)
    paragraph_count = (
        _parse_number(paragraph_match.group(1), default=1)
        if paragraph_match is not None
        else None
    )

    if CONCISE_NOTE_PATTERN.search(request):
        return NoteDetailLevel.CONCISE, paragraph_count or 1

    return NoteDetailLevel.DETAILED, paragraph_count


def _has_explicit_note_preferences(request: str) -> bool:
    """Return whether deterministic note formatting was explicit."""

    return any(
        pattern.search(request) is not None
        for pattern in (
            NOTE_PARAGRAPH_COUNT_PATTERN,
            CONCISE_NOTE_PATTERN,
            DETAILED_NOTE_PATTERN,
        )
    )


def _find_region(request: str) -> str | None:
    """Return the longest matching canonical UK region name."""

    lowered = request.lower()
    for alias in sorted(REGION_ALIASES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}\b", lowered):
            return REGION_ALIASES[alias]
    return None


def _parse_persistence_mode(request: str) -> PersistenceMode:
    """Give explicit write prohibitions precedence over write requests."""

    if PERSISTENCE_FORBIDDEN_PATTERN.search(request):
        return PersistenceMode.FORBIDDEN
    if PERSISTENCE_REQUESTED_PATTERN.search(request):
        return PersistenceMode.REQUESTED
    return PersistenceMode.NOT_REQUESTED


def _requests_latest_data(request: str) -> bool:
    """Detect explicit latest-data wording, defaulting to the latest source data."""

    latest_pattern = re.compile(
        r"\b(?:latest|latest available|most recent|current data)\b",
        re.IGNORECASE,
    )
    return bool(latest_pattern.search(request)) or "as of" not in request.lower()
