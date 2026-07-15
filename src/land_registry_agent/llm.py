"""Bounded OpenAI services for intent interpretation and note drafting."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI, OpenAIError

from land_registry_agent.config import Settings
from land_registry_agent.models import (
    EvidenceBundle,
    Intent,
    ModelDiagnosticResult,
    NoteDetailLevel,
)

INTENT_INSTRUCTIONS = """
You extract a UK property-research request into the supplied schema.

Rules:
- Do not call tools or attempt to retrieve data.
- Treat the deterministic partial interpretation as a starting point.
- Correct a partial value when the full user request clearly contradicts it.
- Explicit prohibitions and negative wording override positive keywords.
- Preserve an explicitly supplied postcode or postcode district.
- Convert a requested trailing period into requested_years.
- Set regional_comparison_requested only when comparison is requested.
- Set regional_comparison_requested to false when comparison is prohibited.
- Use a canonical UK region name when one is identifiable.
- Set street_ranking_requested only when ranking or identifying high-value
  streets is explicitly requested.
- Set street_ranking_requested to false when street ranking is prohibited.
- Set note_requested to false when the user asks to skip, omit, or exclude
  the note, summary, paragraph, report, or write-up.
- Otherwise preserve the partial note_requested value.
- Set note_detail_level to concise for explicitly brief, concise, or short
  notes, and detailed for detailed, comprehensive, or in-depth summaries.
- Set note_paragraph_count only when the user explicitly requests between
  one and five paragraphs; otherwise set it to null.
- Set persistence_mode to:
  - requested when the user asks to save, write, persist, add, record, log,
    store, file, or track the report;
  - forbidden when the user explicitly prohibits writing;
  - not_requested otherwise.
- Explicit write prohibitions override all other persistence wording.
- Do not invent missing postcodes, regions, or optional actions.
- Use clarification_reason when a required value remains ambiguous.
- Confidence must reflect confidence in the interpretation, not data quality.
- Set interpretation_method to llm_assisted.
""".strip()


DRAFT_INSTRUCTIONS = """
Prepare a grounded property-research note from the supplied frozen evidence.

Rules:
- Use only facts, values, dates, street names, and limitations present in
  the evidence.
- Do not calculate new values or infer unsupported conclusions.
- Do not describe local percentage movement when change_claim_permitted
  is false.
- Do not rank streets when street_rankings is empty.
- Do not make a local-versus-regional performance claim when
  regional_comparison_claim_permitted is false.
- When source periods do not overlap, state that comparison is not
  like-for-like.
- State material limitations, including stale HPI data and low confidence.
- Use pounds for price values and percentages for supplied changes.
- Use prose paragraphs without headings, bullets, citations, or JSON.
""".strip()


class LanguageServiceError(RuntimeError):
    """Raised when a bounded language-model operation fails."""


class LanguageModelRefusal(LanguageServiceError):
    """Raised when the model refuses a supported request."""


class NoteDraftingService(Protocol):
    """Interface consumed by the workflow's drafting node."""

    def draft(
        self,
        evidence: EvidenceBundle,
        unsupported_claims: Sequence[str] = (),
    ) -> str:
        """Draft or correct one evidence-grounded research note."""


class OpenAIIntentService:
    """Structured-output fallback used only for ambiguous intent."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self._model = settings.openai_intent_model
        self._client = client or _create_client(settings)

    def interpret(
        self,
        request: str,
        partial_intent: Intent,
    ) -> Intent:
        """Complete an ambiguous deterministic interpretation."""

        input_text = (
            "User request:\n"
            f"{request}\n\n"
            "Deterministic partial interpretation:\n"
            f"{partial_intent.model_dump_json(indent=2)}"
        )

        try:
            response = self._client.responses.parse(
                model=self._model,
                instructions=INTENT_INSTRUCTIONS,
                input=input_text,
                text_format=Intent,
                max_output_tokens=500,
                store=False,
            )
        except OpenAIError as exc:
            raise LanguageServiceError(
                "OpenAI intent interpretation failed: "
                f"{type(exc).__name__}."
            ) from exc

        return _extract_parsed_intent(response)


class OpenAINoteDraftingService:
    """Draft a bounded note from frozen evidence without tool access."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: OpenAI | None = None,
    ) -> None:
        self._model = settings.openai_draft_model
        self._client = client or _create_client(settings)

    def draft(
        self,
        evidence: EvidenceBundle,
        unsupported_claims: Sequence[str] = (),
    ) -> str:
        """Draft an initial note or one verifier-guided correction."""

        evidence_json = json.dumps(
            evidence.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        correction = ""
        if unsupported_claims:
            correction = (
                "\n\nThe previous draft was rejected for these unsupported "
                "claims. Omit or correct them using only the evidence:\n- "
                + "\n- ".join(unsupported_claims)
            )

        try:
            response = self._client.responses.create(
                model=self._model,
                instructions=_draft_instructions(evidence),
                input=f"Frozen evidence:\n{evidence_json}{correction}",
                max_output_tokens=900,
                store=False,
            )
        except OpenAIError as exc:
            raise LanguageServiceError(
                f"OpenAI note drafting failed: {type(exc).__name__}."
            ) from exc

        refusal = _find_refusal(response)
        if refusal is not None:
            raise LanguageModelRefusal(
                "The drafting model refused the research-note request."
            )

        output_text = response.output_text.strip()
        if not output_text:
            raise LanguageServiceError(
                "The drafting model returned no research-note text."
            )

        return _normalise_draft_text(output_text)


def _draft_instructions(evidence: EvidenceBundle) -> str:
    """Build bounded drafting instructions from typed intent."""

    intent = evidence.intent

    if intent.note_paragraph_count is not None:
        paragraph_word = (
            "paragraph" if intent.note_paragraph_count == 1 else "paragraphs"
        )
        format_instruction = (
            f"Write exactly {intent.note_paragraph_count} prose "
            f"{paragraph_word}."
        )
    elif intent.note_detail_level is NoteDetailLevel.CONCISE:
        format_instruction = "Write one concise prose paragraph."
    else:
        format_instruction = (
            "Write a detailed prose summary in two to four short paragraphs."
        )

    return f"{format_instruction}\n\n{DRAFT_INSTRUCTIONS}"


def _normalise_draft_text(output_text: str) -> str:
    """Normalize paragraph whitespace without collapsing paragraph breaks."""

    paragraphs = [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", output_text.strip())
        if paragraph.strip()
    ]
    return "\n\n".join(paragraphs)


def diagnose_model_access(
    settings: Settings,
    *,
    client: OpenAI | None = None,
) -> tuple[ModelDiagnosticResult, ...]:
    """Check configured model metadata without running a generation request."""

    services = (
        ("intent", settings.openai_intent_model),
        ("draft", settings.openai_draft_model),
    )

    if not settings.openai_enabled and client is None:
        return tuple(
            ModelDiagnosticResult(
                service_name=service_name,
                model_name=model_name,
                accessible=False,
                explanation="OPENAI_API_KEY is not configured.",
            )
            for service_name, model_name in services
        )

    openai_client = client or _create_client(settings)
    checked_models: dict[str, tuple[bool, str]] = {}
    results: list[ModelDiagnosticResult] = []

    for service_name, model_name in services:
        if model_name not in checked_models:
            try:
                openai_client.models.retrieve(model_name)
            except OpenAIError as exc:
                checked_models[model_name] = (
                    False,
                    f"Model lookup failed: {type(exc).__name__}.",
                )
            else:
                checked_models[model_name] = (
                    True,
                    "Model metadata is accessible to the configured API key.",
                )

        accessible, explanation = checked_models[model_name]
        results.append(
            ModelDiagnosticResult(
                service_name=service_name,
                model_name=model_name,
                accessible=accessible,
                explanation=explanation,
            )
        )

    return tuple(results)


def _create_client(settings: Settings) -> OpenAI:
    """Create an SDK client without exposing the API key to prompts."""

    if settings.openai_api_key is None:
        raise LanguageServiceError(
            "OPENAI_API_KEY is required for language-model operations."
        )

    return OpenAI(api_key=settings.openai_api_key.get_secret_value())


def _extract_parsed_intent(response: object) -> Intent:
    """Extract a parsed Pydantic value while detecting refusals."""

    for output in getattr(response, "output", ()):
        if getattr(output, "type", None) != "message":
            continue

        for item in getattr(output, "content", ()):
            refusal = getattr(item, "refusal", None)
            if isinstance(refusal, str) and refusal:
                raise LanguageModelRefusal(
                    "The intent model refused the interpretation request."
                )

            parsed = getattr(item, "parsed", None)
            if isinstance(parsed, Intent):
                return parsed
            if parsed is not None:
                return Intent.model_validate(parsed)

    raise LanguageServiceError(
        "The intent model returned no parsed structured output."
    )


def _find_refusal(response: object) -> str | None:
    """Return refusal text from a Responses API result when present."""

    for output in getattr(response, "output", ()):
        for item in getattr(output, "content", ()):
            refusal = getattr(item, "refusal", None)
            if isinstance(refusal, str) and refusal:
                return refusal
    return None


def main() -> int:
    """Run the optional model-access diagnostic."""

    results = diagnose_model_access(Settings())
    print(
        json.dumps(
            [result.model_dump(mode="json") for result in results],
            indent=2,
        )
    )
    return 0 if all(result.accessible for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
