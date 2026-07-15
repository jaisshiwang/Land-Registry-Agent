"""Tests for bounded language-service drafting instructions."""

from __future__ import annotations

from land_registry_agent.llm import (
    _draft_instructions,
    _normalise_draft_text,
)
from land_registry_agent.models import (
    ConfidenceLevel,
    EvidenceBundle,
    Intent,
    InterpretationMethod,
    NoteDetailLevel,
    TrendSummary,
)


def make_evidence(intent: Intent) -> EvidenceBundle:
    """Build minimal frozen evidence for drafting-instruction tests."""

    return EvidenceBundle(
        user_request="Analyse GU1 and prepare a research note.",
        intent=intent,
        source_windows=(),
        monthly_local_metrics=(),
        local_trend=TrendSummary(
            label="GU1 monthly median sale price",
            change_claim_permitted=False,
        ),
        confidence=ConfidenceLevel.LOW,
    )


def make_intent(**updates: object) -> Intent:
    """Build a valid intent with optional note-format updates."""

    intent = Intent(
        postcode="GU1",
        interpretation_method=InterpretationMethod.DETERMINISTIC,
        confidence=0.95,
    )
    return intent.model_copy(update=updates)


def test_detailed_note_defaults_to_two_to_four_paragraphs() -> None:
    instructions = _draft_instructions(make_evidence(make_intent()))

    assert "two to four short paragraphs" in instructions


def test_concise_note_defaults_to_one_paragraph() -> None:
    intent = make_intent(note_detail_level=NoteDetailLevel.CONCISE)

    instructions = _draft_instructions(make_evidence(intent))

    assert "one concise prose paragraph" in instructions


def test_explicit_paragraph_count_overrides_detail_default() -> None:
    intent = make_intent(
        note_detail_level=NoteDetailLevel.DETAILED,
        note_paragraph_count=3,
    )

    instructions = _draft_instructions(make_evidence(intent))

    assert "exactly 3 prose paragraphs" in instructions
    assert "two to four short paragraphs" not in instructions


def test_draft_normalisation_preserves_paragraph_breaks() -> None:
    output = " First   paragraph.\n\n Second\nparagraph. "

    assert _normalise_draft_text(output) == (
        "First paragraph.\n\nSecond paragraph."
    )
