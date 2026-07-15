"""Tests for deterministic parsing, fallback interpretation and policy."""

from __future__ import annotations

from land_registry_agent.intent import interpret_request, parse_deterministically
from land_registry_agent.models import (
    Intent,
    InterpretationMethod,
    NoteDetailLevel,
    PersistenceMode,
)


class StubIntentFallback:
    """Deterministic substitute for the structured-output LLM service."""

    def __init__(self, resolved_postcode: str = "SW1A") -> None:
        self.resolved_postcode = resolved_postcode
        self.calls: list[tuple[str, Intent]] = []

    def interpret(
        self,
        request: str,
        partial_intent: Intent,
    ) -> Intent:
        self.calls.append((request, partial_intent))
        return partial_intent.model_copy(
            update={
                "postcode": self.resolved_postcode,
                "confidence": 0.90,
                "clarification_reason": None,
            }
        )


class UnexpectedFallback:
    """Fail if a complete deterministic request invokes the fallback."""

    def interpret(
        self,
        request: str,
        partial_intent: Intent,
    ) -> Intent:
        raise AssertionError(
            f"Fallback unexpectedly called for {request!r}: {partial_intent}"
        )


class OptionalActionFallback:
    """Resolve optional-action ambiguity without making a model call."""

    def __init__(self, **updates: object) -> None:
        self.updates = updates
        self.calls: list[tuple[str, Intent]] = []

    def interpret(
        self,
        request: str,
        partial_intent: Intent,
    ) -> Intent:
        self.calls.append((request, partial_intent))
        return partial_intent.model_copy(
            update={
                **self.updates,
                "confidence": 0.90,
                "clarification_reason": None,
            }
        )


def test_parses_target_request_deterministically() -> None:
    request = (
        "Analyse property price trends in GU1 over the last 3 years. "
        "Compare with the South East regional average. Identify the "
        "highest-value streets. Then prepare a one-paragraph research note "
        "and add it to my tracking sheet."
    )

    intent = interpret_request(
        request,
        fallback=UnexpectedFallback(),
    )

    assert intent.postcode == "GU1"
    assert intent.requested_years == 3
    assert intent.region == "South East"
    assert intent.regional_comparison_requested is True
    assert intent.street_ranking_requested is True
    assert intent.note_requested is True
    assert intent.note_detail_level is NoteDetailLevel.DETAILED
    assert intent.note_paragraph_count == 1
    assert intent.persistence_mode is PersistenceMode.REQUESTED
    assert intent.latest_available_data is True
    assert intent.interpretation_method is InterpretationMethod.DETERMINISTIC
    assert intent.clarification_reason is None


def test_explicit_write_prohibition_overrides_persistence_wording() -> None:
    intent = parse_deterministically(
        "Analyse GU1 for two years and add a note, but do not save it."
    )

    assert intent.postcode == "GU1"
    assert intent.requested_years == 2
    assert intent.note_requested is True
    assert intent.persistence_mode is PersistenceMode.FORBIDDEN


def test_analysis_only_omits_note_and_persistence() -> None:
    intent = interpret_request(
        "Analyse prices in GU1 over five years, analysis only.",
        fallback=UnexpectedFallback(),
    )

    assert intent.requested_years == 5
    assert intent.note_requested is False
    assert intent.persistence_mode is PersistenceMode.NOT_REQUESTED
    assert intent.clarification_reason is None


def test_ambiguous_request_uses_structured_fallback() -> None:
    fallback = StubIntentFallback()

    intent = interpret_request(
        "Compare property prices over four years with London.",
        fallback=fallback,
    )

    assert len(fallback.calls) == 1

    request, partial_intent = fallback.calls[0]
    assert request == "Compare property prices over four years with London."
    assert partial_intent.postcode is None
    assert partial_intent.region == "London"
    assert partial_intent.regional_comparison_requested is True

    assert intent.postcode == "SW1A"
    assert intent.requested_years == 4
    assert intent.region == "London"
    assert intent.interpretation_method is InterpretationMethod.LLM_ASSISTED
    assert intent.confidence == 0.90
    assert intent.clarification_reason is None


def test_missing_postcode_requests_clarification_without_fallback() -> None:
    intent = interpret_request(
        "Analyse property prices over three years.",
        fallback=None,
    )

    assert intent.postcode is None
    assert intent.confidence < 0.5
    assert intent.clarification_reason is not None
    assert "postcode" in intent.clarification_reason.lower()


def test_persistence_without_a_note_requires_clarification() -> None:
    intent = interpret_request(
        "Analyse GU1 for three years, analysis only, then save it.",
        fallback=UnexpectedFallback(),
    )

    assert intent.note_requested is False
    assert intent.persistence_mode is PersistenceMode.REQUESTED
    assert intent.clarification_reason is not None
    assert "requires a research note" in intent.clarification_reason.lower()


def test_live_subscription_request_is_rejected_by_bounded_policy() -> None:
    intent = interpret_request(
        "Analyse GU1 for three years and subscribe to live updates.",
        fallback=UnexpectedFallback(),
    )

    assert intent.live_refresh_requested is True
    assert intent.clarification_reason is not None
    assert "outside the demonstration workflow" in intent.clarification_reason


def test_tracker_wording_uses_fallback_for_persistence() -> None:
    fallback = OptionalActionFallback(
        persistence_mode=PersistenceMode.REQUESTED,
        note_requested=False,
        region="South East",
    )

    intent = interpret_request(
        "Analyse GU1 for three years and record it in my property tracker.",
        fallback=fallback,
    )

    assert len(fallback.calls) == 1
    assert intent.persistence_mode is PersistenceMode.REQUESTED
    assert intent.note_requested is True
    assert intent.region is None
    assert intent.interpretation_method is InterpretationMethod.LLM_ASSISTED
    assert intent.clarification_reason is None


def test_priciest_roads_wording_uses_fallback_for_street_ranking() -> None:
    fallback = OptionalActionFallback(street_ranking_requested=True)

    intent = interpret_request(
        "Analyse GU1 for three years and show me the priciest roads.",
        fallback=fallback,
    )

    assert len(fallback.calls) == 1
    assert intent.street_ranking_requested is True


def test_negated_street_ranking_uses_fallback_to_correct_partial() -> None:
    fallback = OptionalActionFallback(street_ranking_requested=False)

    intent = interpret_request(
        "Analyse GU1 for three years but do not rank streets.",
        fallback=fallback,
    )

    assert len(fallback.calls) == 1
    assert fallback.calls[0][1].street_ranking_requested is True
    assert intent.street_ranking_requested is False


def test_summary_omission_uses_fallback() -> None:
    fallback = OptionalActionFallback(
        note_requested=False,
        note_paragraph_count=3,
    )

    intent = interpret_request(
        "Analyse GU1 for three years and leave out the written summary.",
        fallback=fallback,
    )

    assert len(fallback.calls) == 1
    assert intent.note_requested is False
    assert intent.note_paragraph_count is None


def test_comparison_rephrasing_uses_fallback() -> None:
    fallback = OptionalActionFallback(
        regional_comparison_requested=True
    )

    intent = interpret_request(
        "Analyse GU1 for three years against the wider South East.",
        fallback=fallback,
    )

    assert len(fallback.calls) == 1
    assert intent.region == "South East"
    assert intent.regional_comparison_requested is True


def test_parses_brief_summary_as_one_concise_paragraph() -> None:
    intent = parse_deterministically(
        "Analyse GU1 for three years and prepare a brief summary."
    )

    assert intent.note_detail_level is NoteDetailLevel.CONCISE
    assert intent.note_paragraph_count == 1


def test_parses_detailed_summary_without_exact_count() -> None:
    intent = parse_deterministically(
        "Analyse GU1 for three years and prepare a detailed summary."
    )

    assert intent.note_detail_level is NoteDetailLevel.DETAILED
    assert intent.note_paragraph_count is None


def test_parses_explicit_paragraph_count() -> None:
    intent = parse_deterministically(
        "Analyse GU1 and prepare a three-paragraph report."
    )

    assert intent.note_detail_level is NoteDetailLevel.DETAILED
    assert intent.note_paragraph_count == 3
