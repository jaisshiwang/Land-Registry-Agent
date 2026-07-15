"""Tests for approved-report persistence and owner scoping."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from land_registry_agent.analysis import content_hash
from land_registry_agent.models import (
    ApprovalChoice,
    ApprovalDecision,
    AuditEvent,
    AuditStatus,
    ConfidenceLevel,
    EvidenceBundle,
    Intent,
    InterpretationMethod,
    MonthlyMetric,
    PersistenceMode,
    ReportPayload,
    SourceWindow,
    TrendSummary,
    UserContext,
)
from land_registry_agent.repository import (
    ApprovalMismatchError,
    RepositoryError,
    SQLiteReportRepository,
)


def make_evidence() -> EvidenceBundle:
    """Build minimal valid evidence for persistence tests."""

    intent = Intent(
        postcode="GU1",
        requested_years=3,
        regional_comparison_requested=False,
        street_ranking_requested=False,
        note_requested=True,
        persistence_mode=PersistenceMode.REQUESTED,
        latest_available_data=True,
        live_refresh_requested=False,
        interpretation_method=InterpretationMethod.DETERMINISTIC,
        confidence=0.95,
    )

    return EvidenceBundle(
        user_request="Analyse GU1 and save the note.",
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
        confidence=ConfidenceLevel.HIGH,
        source_urls=("https://example.test/price-paid",),
        artifact_keys=("artifact-1",),
    )


def make_payload(
    *,
    run_id: str = "run-001",
    owner_id: str = "owner-001",
    owner_name: str = "Owner One",
    report_name: str = "GU1 Property Trends",
) -> ReportPayload:
    """Build a payload with a valid evidence hash."""

    evidence = make_evidence()
    return ReportPayload(
        run_id=run_id,
        owner=UserContext(
            user_id=owner_id,
            display_name=owner_name,
        ),
        report_name=report_name,
        destination="SQLite tracking sheet",
        user_request=evidence.user_request,
        postcode="GU1",
        requested_years=3,
        research_note="GU1 property evidence was analysed.",
        charts=(),
        evidence=evidence,
        evidence_hash=content_hash(evidence),
    )


def make_approval(
    payload: ReportPayload,
    choice: ApprovalChoice = ApprovalChoice.APPROVE,
) -> ApprovalDecision:
    """Build a hash-bound approval decision."""

    return ApprovalDecision(
        run_id=payload.run_id,
        owner_id=payload.owner.user_id,
        report_hash=content_hash(payload),
        choice=choice,
    )


def make_trace() -> tuple[AuditEvent, ...]:
    """Build an ordered pre-write audit trace."""

    return (
        AuditEvent(
            sequence=1,
            action="prepare_report",
            status=AuditStatus.SUCCEEDED,
            explanation="Prepared the exact report payload.",
        ),
        AuditEvent(
            sequence=2,
            action="request_approval",
            status=AuditStatus.SUCCEEDED,
            explanation="The owner approved the exact report hash.",
        ),
    )


@pytest.fixture
def repository(tmp_path: Path) -> SQLiteReportRepository:
    """Return an isolated initialized repository."""

    result = SQLiteReportRepository(tmp_path / "reports.sqlite3")
    result.initialize()
    return result


def test_successful_write_generates_unique_report_id(
    repository: SQLiteReportRepository,
) -> None:
    first_payload = make_payload(run_id="run-001")
    second_payload = make_payload(
        run_id="run-002",
        report_name="GU1 Property Trends Two",
    )

    first = repository.save_approved(
        first_payload,
        make_approval(first_payload),
        make_trace(),
    )
    second = repository.save_approved(
        second_payload,
        make_approval(second_payload),
        make_trace(),
    )

    assert first.report_id.startswith("rpt_")
    assert second.report_id.startswith("rpt_")
    assert first.report_id != second.report_id
    assert first.idempotent_replay is False
    assert second.idempotent_replay is False


def test_repeated_write_is_idempotent(
    repository: SQLiteReportRepository,
) -> None:
    payload = make_payload()

    first = repository.save_approved(
        payload,
        make_approval(payload),
        make_trace(),
    )
    replay = repository.save_approved(
        payload,
        make_approval(payload),
        make_trace(),
    )

    assert replay.report_id == first.report_id
    assert replay.report_hash == first.report_hash
    assert replay.idempotent_replay is True
    assert len(repository.list_for_owner("owner-001")) == 1


def test_report_listing_and_reads_are_owner_scoped(
    repository: SQLiteReportRepository,
) -> None:
    owner_one_payload = make_payload(
        run_id="run-owner-one",
        owner_id="owner-001",
        owner_name="Owner One",
    )
    owner_two_payload = make_payload(
        run_id="run-owner-two",
        owner_id="owner-002",
        owner_name="Owner Two",
    )

    owner_one_report = repository.save_approved(
        owner_one_payload,
        make_approval(owner_one_payload),
        make_trace(),
    )
    owner_two_report = repository.save_approved(
        owner_two_payload,
        make_approval(owner_two_payload),
        make_trace(),
    )

    owner_one_results = repository.list_for_owner("owner-001")
    owner_two_results = repository.list_for_owner("owner-002")

    assert [item.report_id for item in owner_one_results] == [
        owner_one_report.report_id
    ]
    assert [item.report_id for item in owner_two_results] == [
        owner_two_report.report_id
    ]

    assert (
        repository.get_for_owner(
            owner_one_report.report_id,
            "owner-001",
        )
        is not None
    )
    assert (
        repository.get_for_owner(
            owner_one_report.report_id,
            "owner-002",
        )
        is None
    )


def test_stale_report_hash_is_rejected_without_write(
    repository: SQLiteReportRepository,
) -> None:
    payload = make_payload()
    stale_approval = make_approval(payload).model_copy(
        update={"report_hash": "0" * 64}
    )

    with pytest.raises(
        ApprovalMismatchError,
        match="does not match",
    ):
        repository.save_approved(
            payload,
            stale_approval,
            make_trace(),
        )

    assert repository.list_for_owner("owner-001") == ()


def test_rejected_decision_cannot_be_persisted(
    repository: SQLiteReportRepository,
) -> None:
    payload = make_payload()

    with pytest.raises(
        ApprovalMismatchError,
        match="rejected report",
    ):
        repository.save_approved(
            payload,
            make_approval(payload, ApprovalChoice.REJECT),
            make_trace(),
        )

    assert repository.list_for_owner("owner-001") == ()


def test_audit_trace_must_be_contiguous(
    repository: SQLiteReportRepository,
) -> None:
    payload = make_payload()
    invalid_trace = (
        AuditEvent(
            sequence=2,
            action="request_approval",
            status=AuditStatus.SUCCEEDED,
            explanation="Sequence one is missing.",
        ),
    )

    with pytest.raises(
        RepositoryError,
        match="contiguous and ordered",
    ):
        repository.save_approved(
            payload,
            make_approval(payload),
            invalid_trace,
        )
