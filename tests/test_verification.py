"""Tests for evidence hashing and deterministic note verification."""

from __future__ import annotations

from datetime import date

from land_registry_agent.analysis import content_hash, verify_note
from land_registry_agent.models import (
    ConfidenceLevel,
    EvidenceBundle,
    Intent,
    InterpretationMethod,
    MonthlyMetric,
    PersistenceMode,
    SourceWindow,
    StreetMetric,
    TrendSummary,
)


def make_evidence() -> EvidenceBundle:
    """Build compact evidence containing an allowlist of supported claims."""

    intent = Intent(
        postcode="GU1",
        requested_years=3,
        region="South East",
        regional_comparison_requested=True,
        street_ranking_requested=True,
        note_requested=True,
        persistence_mode=PersistenceMode.REQUESTED,
        latest_available_data=True,
        live_refresh_requested=False,
        interpretation_method=InterpretationMethod.DETERMINISTIC,
        confidence=0.95,
    )

    return EvidenceBundle(
        user_request="Analyse GU1 and save a report.",
        intent=intent,
        source_windows=(
            SourceWindow(
                source_name="HM Land Registry Price Paid Data",
                start_date=date(2024, 1, 1),
                end_date=date(2024, 2, 1),
                latest_available_date=date(2024, 2, 1),
            ),
        ),
        monthly_local_metrics=(
            MonthlyMetric(
                period=date(2024, 1, 1),
                transaction_count=5,
                median_price_gbp=200_000,
                mean_price_gbp=205_000,
            ),
            MonthlyMetric(
                period=date(2024, 2, 1),
                transaction_count=5,
                median_price_gbp=220_000,
                mean_price_gbp=218_000,
            ),
        ),
        local_trend=TrendSummary(
            label="GU1 monthly median sale price",
            start_value=200_000,
            end_value=220_000,
            percentage_change=10.0,
            change_claim_permitted=True,
        ),
        regional_trend=TrendSummary(
            label="South East regional average price",
            start_value=300_000,
            end_value=315_000,
            percentage_change=5.0,
            change_claim_permitted=True,
        ),
        periods_overlap=True,
        regional_comparison_claim_permitted=True,
        comparison_difference_percentage_points=5.0,
        street_rankings=(
            StreetMetric(
                rank=1,
                street="High Street",
                median_price_gbp=500_000,
                transaction_count=3,
            ),
        ),
        confidence=ConfidenceLevel.HIGH,
        limitations=(),
        charts=(),
        source_urls=("https://example.test/price-paid",),
        artifact_keys=("artifact-1",),
    )


def test_content_hash_is_deterministic_for_equivalent_evidence() -> None:
    evidence = make_evidence()
    reconstructed = EvidenceBundle.model_validate(
        evidence.model_dump(mode="json")
    )

    assert content_hash(evidence) == content_hash(reconstructed)
    assert len(content_hash(evidence)) == 64


def test_content_hash_changes_when_evidence_changes() -> None:
    evidence = make_evidence()
    changed = evidence.model_copy(
        update={"limitations": ("A new limitation was added.",)}
    )

    assert content_hash(evidence) != content_hash(changed)


def test_verifier_accepts_supported_dates_numbers_and_street_claim() -> None:
    evidence = make_evidence()
    note = (
        "From January 2024 to February 2024, the local median rose "
        "from £200,000 to £220,000, a 10% change, while High Street "
        "was the highest-value street at £500,000 from 3 transactions."
    )

    result = verify_note(note, evidence)

    assert result.supported is True
    assert result.checked_claim_count > 0
    assert result.unsupported_claims == ()


def test_verifier_accepts_unsigned_magnitude_for_negative_percentage() -> None:
    evidence = make_evidence().model_copy(
        update={
            "local_trend": TrendSummary(
                label="GU1 monthly median sale price",
                start_value=220_000,
                end_value=198_000,
                percentage_change=-10.0,
                change_claim_permitted=True,
            ),
        }
    )

    result = verify_note(
        "The local median price recorded a decline of 10%.",
        evidence,
    )

    assert result.supported is True
    assert result.unsupported_claims == ()


def test_verifier_rejects_unsupported_number_and_date() -> None:
    result = verify_note(
        "In 2025-01 the median price reached £999,999.",
        make_evidence(),
    )

    assert result.supported is False
    assert any(
        claim == "Unsupported date: 2025-01"
        for claim in result.unsupported_claims
    )
    assert any(
        claim == "Unsupported number: £999,999"
        for claim in result.unsupported_claims
    )


def test_verifier_rejects_performance_claim_without_comparison() -> None:
    evidence = make_evidence().model_copy(
        update={
            "periods_overlap": False,
            "regional_comparison_claim_permitted": False,
            "comparison_difference_percentage_points": None,
        }
    )

    result = verify_note(
        "Local prices outperformed the regional trend.",
        evidence,
    )

    assert result.supported is False
    assert (
        "Unsupported local-versus-regional performance claim."
        in result.unsupported_claims
    )


def test_verifier_rejects_street_superlative_without_rankings() -> None:
    evidence = make_evidence().model_copy(
        update={"street_rankings": ()}
    )

    result = verify_note(
        "High Street was the highest-value street.",
        evidence,
    )

    assert result.supported is False
    assert "Unsupported street-ranking claim." in result.unsupported_claims
