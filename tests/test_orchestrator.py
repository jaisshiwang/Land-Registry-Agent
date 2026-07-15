"""Tests for LangGraph orchestration and the human write gate."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from land_registry_agent.analysis import PropertyAnalyzer
from land_registry_agent.config import Settings
from land_registry_agent.data import FetchResult
from land_registry_agent.models import (
    ApprovalChoice,
    ApprovalDecision,
    EvidenceBundle,
    HPIRecord,
    StepStatus,
    Transaction,
    UserContext,
)
from land_registry_agent.orchestrator import Orchestrator, WorkflowState
from land_registry_agent.repository import SQLiteReportRepository


class RecordingGateway:
    """Network-free gateway that records every requested read."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch_price_paid_transactions(
        self,
        postcode: str,
        requested_years: int,
    ) -> FetchResult[Transaction]:
        self.calls.append("fetch_price_paid_transactions")
        return FetchResult(
            records=make_transactions(),
            source_url="https://example.test/price-paid",
            artifact_keys=("price-artifact",),
        )

    def fetch_regional_hpi(
        self,
        region: str,
        requested_years: int,
    ) -> FetchResult[HPIRecord]:
        self.calls.append("fetch_regional_hpi")
        return FetchResult(
            records=(
                HPIRecord(
                    period=date(2016, 1, 1),
                    region=region,
                    average_price_gbp=300_000,
                ),
                HPIRecord(
                    period=date(2016, 2, 1),
                    region=region,
                    average_price_gbp=303_000,
                ),
            ),
            source_url="https://example.test/hpi",
            artifact_keys=("hpi-artifact",),
        )


class GroundedDraftingService:
    """Return a claim-free paragraph that passes verification."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def draft(
        self,
        evidence: EvidenceBundle,
        unsupported_claims: Sequence[str] = (),
    ) -> str:
        self.calls.append(tuple(unsupported_claims))
        return (
            "The available property evidence was analysed, subject to the "
            "limitations recorded in the evidence bundle."
        )


class CorrectiveDraftingService:
    """Return one unsupported draft followed by a supported correction."""

    def __init__(self) -> None:
        self.calls = 0

    def draft(
        self,
        evidence: EvidenceBundle,
        unsupported_claims: Sequence[str] = (),
    ) -> str:
        self.calls += 1
        if self.calls == 1:
            return "The local median price was £999,999."
        return (
            "The available property evidence was analysed, subject to the "
            "limitations recorded in the evidence bundle."
        )


@dataclass
class Harness:
    """Isolated graph and dependencies used by one test."""

    orchestrator: Orchestrator
    gateway: RecordingGateway
    repository: SQLiteReportRepository
    drafting_service: GroundedDraftingService | CorrectiveDraftingService


def make_transactions() -> tuple[Transaction, ...]:
    """Return sufficient local evidence over two months."""

    return tuple(
        Transaction(
            transaction_id=f"transaction-{number}",
            transfer_date=date(
                2024,
                1 if number <= 5 else 2,
                number if number <= 5 else number - 5,
            ),
            price_gbp=200_000 + number * 5_000,
            postcode="GU1 1AA",
            property_type="terraced",
            street=(
                "High Street"
                if number <= 5
                else "Lower Street"
            ),
        )
        for number in range(1, 11)
    )


def build_harness(
    tmp_path: Path,
    drafting_service: (
        GroundedDraftingService | CorrectiveDraftingService
    ),
) -> Harness:
    """Build an orchestrator using only local deterministic fakes."""

    settings = Settings(
        checkpoint_db_path=tmp_path / "checkpoints.sqlite3",
        reports_db_path=tmp_path / "reports.sqlite3",
        cache_directory=tmp_path / "cache",
        minimum_local_transactions=10,
        minimum_street_transactions=3,
        maximum_corrective_redrafts=1,
    )
    gateway = RecordingGateway()
    repository = SQLiteReportRepository(settings.reports_db_path)
    orchestrator = Orchestrator(
        settings=settings,
        gateway=gateway,
        analyzer=PropertyAnalyzer(settings),
        drafting_service=drafting_service,
        repository=repository,
        intent_fallback=None,
    )
    orchestrator.initialise()

    return Harness(
        orchestrator=orchestrator,
        gateway=gateway,
        repository=repository,
        drafting_service=drafting_service,
    )


@pytest.fixture
def harness(tmp_path: Path) -> Iterator[Harness]:
    """Yield a graph with isolated checkpoint and report databases."""

    result = build_harness(
        tmp_path,
        GroundedDraftingService(),
    )
    try:
        yield result
    finally:
        result.orchestrator.close()


def run_to_approval(harness: Harness) -> WorkflowState:
    """Prepare and execute a persistence request up to its interrupt."""

    prepared = harness.orchestrator.prepare(
        user_request=(
            "Analyse GU1 over three years, prepare a research note "
            "and save it."
        ),
        user=UserContext(
            user_id="owner-001",
            display_name="Owner One",
        ),
        report_name="GU1 Test Report",
        run_id="run-approval",
    )
    assert prepared["status"] == "planned"

    return harness.orchestrator.execute(prepared["run_id"])


def test_plan_is_checkpointed_before_any_external_read(
    harness: Harness,
) -> None:
    prepared = harness.orchestrator.prepare(
        user_request="Analyse GU1 over three years, analysis only.",
        user=UserContext(
            user_id="owner-001",
            display_name="Owner One",
        ),
        run_id="run-plan-first",
    )

    assert prepared["status"] == "planned"
    assert prepared["plan"].run_id == "run-plan-first"
    assert harness.gateway.calls == []

    completed = harness.orchestrator.execute("run-plan-first")

    assert completed["status"] == "completed"
    assert harness.gateway.calls == ["fetch_price_paid_transactions"]


def test_persistence_pauses_before_repository_write(
    harness: Harness,
) -> None:
    paused = run_to_approval(harness)

    assert paused["status"] == "awaiting_approval"
    assert paused["report_hash"]
    assert paused["report_payload"].report_name == "GU1 Test Report"
    assert harness.repository.list_for_owner("owner-001") == ()

    save_step = next(
        step
        for step in paused["plan"].steps
        if step.operation == "save_approved_report"
    )
    assert save_step.status is StepStatus.PENDING

    assert paused["audit_trace"][-1].action == "request_approval"
    assert paused["audit_trace"][-1].status.value == "paused"


def test_rejection_completes_without_write(
    harness: Harness,
) -> None:
    paused = run_to_approval(harness)
    decision = ApprovalDecision(
        run_id=paused["run_id"],
        owner_id=paused["user"].user_id,
        report_hash=paused["report_hash"],
        choice=ApprovalChoice.REJECT,
    )

    rejected = harness.orchestrator.resume_approval(
        run_id=paused["run_id"],
        decision=decision,
    )

    assert rejected["status"] == "rejected"
    assert "saved_report" not in rejected
    assert harness.repository.list_for_owner("owner-001") == ()

    save_step = next(
        step
        for step in rejected["plan"].steps
        if step.operation == "save_approved_report"
    )
    assert save_step.status is StepStatus.SKIPPED


def test_stale_approval_hash_is_rejected(
    harness: Harness,
) -> None:
    paused = run_to_approval(harness)
    stale_decision = ApprovalDecision(
        run_id=paused["run_id"],
        owner_id=paused["user"].user_id,
        report_hash="0" * 64,
        choice=ApprovalChoice.APPROVE,
    )

    failed = harness.orchestrator.resume_approval(
        run_id=paused["run_id"],
        decision=stale_decision,
    )

    assert failed["status"] == "failed"
    assert failed["error"] is not None
    assert "stale or different report" in failed["error"]
    assert harness.repository.list_for_owner("owner-001") == ()


def test_approved_payload_is_saved_exactly_once(
    harness: Harness,
) -> None:
    paused = run_to_approval(harness)
    decision = ApprovalDecision(
        run_id=paused["run_id"],
        owner_id=paused["user"].user_id,
        report_hash=paused["report_hash"],
        choice=ApprovalChoice.APPROVE,
    )

    completed = harness.orchestrator.resume_approval(
        run_id=paused["run_id"],
        decision=decision,
    )

    assert completed["status"] == "completed"

    saved = completed["saved_report"]
    assert saved.report_hash == paused["report_hash"]
    assert saved.payload == paused["report_payload"]
    assert saved.payload.research_note == paused["draft"]

    owner_reports = harness.repository.list_for_owner("owner-001")
    assert len(owner_reports) == 1
    assert owner_reports[0].report_id == saved.report_id

    sequences = [
        event.sequence
        for event in completed["audit_trace"]
    ]
    assert sequences == list(range(1, len(sequences) + 1))


def test_one_corrective_redraft_is_allowed(
    tmp_path: Path,
) -> None:
    drafting_service = CorrectiveDraftingService()
    harness = build_harness(tmp_path, drafting_service)

    try:
        prepared = harness.orchestrator.prepare(
            user_request=(
                "Analyse GU1 over three years and prepare a research note."
            ),
            user=UserContext(
                user_id="owner-001",
                display_name="Owner One",
            ),
            run_id="run-redraft",
        )
        completed = harness.orchestrator.execute(prepared["run_id"])
    finally:
        harness.orchestrator.close()

    assert completed["status"] == "completed"
    assert completed["verification"].supported is True
    assert completed["draft_attempts"] == 2
    assert drafting_service.calls == 2
    assert "£999,999" not in completed["draft"]
