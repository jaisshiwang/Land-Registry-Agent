"""Owner-scoped SQLite persistence for explicitly approved reports."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ValidationError

from land_registry_agent.analysis import canonical_json, content_hash
from land_registry_agent.models import (
    ApprovalChoice,
    ApprovalDecision,
    AuditEvent,
    AuditStatus,
    ReportPayload,
    SavedReport,
)


class RepositoryError(RuntimeError):
    """Raised when approved-report persistence fails safely."""


class ApprovalMismatchError(RepositoryError):
    """Raised when approval does not match the proposed report."""


class OwnershipConflictError(RepositoryError):
    """Raised when an idempotency key belongs to another owner."""


class ApprovedReportRepository(Protocol):
    """Persistence interface consumed by the workflow."""

    def initialize(self) -> None:
        """Create the approved-report schema if required."""

    def save_approved(
        self,
        payload: ReportPayload,
        approval: ApprovalDecision,
        audit_trace: Sequence[AuditEvent],
    ) -> SavedReport:
        """Idempotently save an exact, explicitly approved payload."""

    def list_for_owner(self, owner_id: str) -> tuple[SavedReport, ...]:
        """List reports belonging to one owner."""

    def get_for_owner(
        self,
        report_id: str,
        owner_id: str,
    ) -> SavedReport | None:
        """Retrieve one report using report ID and owner ID together."""


class SQLiteReportRepository:
    """SQLite implementation with owner scoping and idempotent writes."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    def initialize(self) -> None:
        """Create the database directory and approved-report schema."""

        self._database_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connection() as connection, connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS approved_reports (
                        report_id TEXT PRIMARY KEY,
                        owner_id TEXT NOT NULL,
                        owner_display_name TEXT NOT NULL,
                        report_name TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        approved_at TEXT NOT NULL,
                        user_request TEXT NOT NULL,
                        postcode TEXT NOT NULL,
                        region TEXT,
                        requested_years INTEGER NOT NULL,
                        research_note TEXT NOT NULL,
                        chart_data_json TEXT NOT NULL,
                        evidence_json TEXT NOT NULL,
                        trace_json TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        evidence_hash TEXT NOT NULL,
                        report_hash TEXT NOT NULL,
                        approval_status TEXT NOT NULL
                            CHECK (approval_status = 'approved'),
                        UNIQUE (run_id, report_hash)
                    )
                    """
                )
                connection.execute(
                    """
                    CREATE INDEX IF NOT EXISTS
                        idx_approved_reports_owner_approved
                    ON approved_reports (owner_id, approved_at DESC)
                    """
                )

    def save_approved(
        self,
        payload: ReportPayload,
        approval: ApprovalDecision,
        audit_trace: Sequence[AuditEvent],
    ) -> SavedReport:
        """Validate and idempotently persist the exact approved payload."""

        report_hash = self._validate_approval(payload, approval)
        self._validate_audit_trace(audit_trace)

        with self._connection() as connection, connection:
                existing = connection.execute(
                    """
                    SELECT *
                    FROM approved_reports
                    WHERE run_id = ? AND report_hash = ?
                    """,
                    (payload.run_id, report_hash),
                ).fetchone()

                if existing is not None:
                    if existing["owner_id"] != payload.owner.user_id:
                        raise OwnershipConflictError(
                            "The idempotency key belongs to another owner."
                        )
                    return self._row_to_report(
                        existing,
                        idempotent_replay=True,
                    )

                report_id = f"rpt_{uuid4().hex}"
                saved_event = AuditEvent(
                    sequence=len(audit_trace) + 1,
                    action="save_report",
                    status=AuditStatus.SUCCEEDED,
                    explanation=(
                        f"Saved approved report {report_id} for owner "
                        f"{payload.owner.user_id}."
                    ),
                    metadata={
                        "report_id": report_id,
                        "owner_id": payload.owner.user_id,
                    },
                )
                stored_trace = (*audit_trace, saved_event)

                try:
                    connection.execute(
                        """
                        INSERT INTO approved_reports (
                            report_id,
                            owner_id,
                            owner_display_name,
                            report_name,
                            run_id,
                            created_at,
                            approved_at,
                            user_request,
                            postcode,
                            region,
                            requested_years,
                            research_note,
                            chart_data_json,
                            evidence_json,
                            trace_json,
                            payload_json,
                            evidence_hash,
                            report_hash,
                            approval_status
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, 'approved')
                        """,
                        (
                            report_id,
                            payload.owner.user_id,
                            payload.owner.display_name,
                            payload.report_name,
                            payload.run_id,
                            payload.created_at.isoformat(),
                            approval.decided_at.isoformat(),
                            payload.user_request,
                            payload.postcode,
                            payload.region,
                            payload.requested_years,
                            payload.research_note,
                            _models_json(payload.charts),
                            canonical_json(payload.evidence),
                            _models_json(stored_trace),
                            canonical_json(payload),
                            payload.evidence_hash,
                            report_hash,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise RepositoryError(
                        "Approved report insert violated a database constraint."
                    ) from exc

                row = connection.execute(
                    """
                    SELECT *
                    FROM approved_reports
                    WHERE report_id = ? AND owner_id = ?
                    """,
                    (report_id, payload.owner.user_id),
                ).fetchone()

                if row is None:
                    raise RepositoryError(
                        "Saved report could not be read back."
                    )

                return self._row_to_report(
                    row,
                    idempotent_replay=False,
                )

    def list_for_owner(self, owner_id: str) -> tuple[SavedReport, ...]:
        """List only reports belonging to the supplied owner."""

        owner_id = _required_identity(owner_id)

        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM approved_reports
                WHERE owner_id = ?
                ORDER BY approved_at DESC
                """,
                (owner_id,),
            ).fetchall()

        return tuple(self._row_to_report(row) for row in rows)

    def get_for_owner(
        self,
        report_id: str,
        owner_id: str,
    ) -> SavedReport | None:
        """Retrieve a report only when both identifiers match."""

        report_id = _required_identity(report_id)
        owner_id = _required_identity(owner_id)

        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM approved_reports
                WHERE report_id = ? AND owner_id = ?
                """,
                (report_id, owner_id),
            ).fetchone()

        return self._row_to_report(row) if row is not None else None

    def _validate_approval(
        self,
        payload: ReportPayload,
        approval: ApprovalDecision,
    ) -> str:
        """Require an approved decision matching the exact payload."""

        if approval.choice is not ApprovalChoice.APPROVE:
            raise ApprovalMismatchError(
                "A rejected report cannot be persisted."
            )

        if approval.run_id != payload.run_id:
            raise ApprovalMismatchError(
                "Approval run ID does not match the report."
            )

        if approval.owner_id != payload.owner.user_id:
            raise ApprovalMismatchError(
                "Approval owner does not match the report owner."
            )

        calculated_evidence_hash = content_hash(payload.evidence)
        if calculated_evidence_hash != payload.evidence_hash:
            raise ApprovalMismatchError(
                "Evidence changed after its hash was calculated."
            )

        calculated_report_hash = content_hash(payload)
        if calculated_report_hash != approval.report_hash:
            raise ApprovalMismatchError(
                "Approval hash does not match the exact report payload."
            )

        return calculated_report_hash

    def _validate_audit_trace(
        self,
        audit_trace: Sequence[AuditEvent],
    ) -> None:
        """Require a contiguous, correctly ordered audit trace."""

        actual = [event.sequence for event in audit_trace]
        expected = list(range(1, len(audit_trace) + 1))
        if actual != expected:
            raise RepositoryError(
                "Audit event sequence must be contiguous and ordered."
            )

    def _row_to_report(
        self,
        row: sqlite3.Row,
        *,
        idempotent_replay: bool = False,
    ) -> SavedReport:
        """Deserialize and integrity-check one stored report."""

        try:
            payload = ReportPayload.model_validate_json(row["payload_json"])
            trace_items = json.loads(row["trace_json"])
            audit_trace = tuple(
                AuditEvent.model_validate(item) for item in trace_items
            )
            approved_at = datetime.fromisoformat(row["approved_at"])
        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            ValidationError,
        ) as exc:
            raise RepositoryError(
                "Stored report data is malformed."
            ) from exc

        if payload.owner.user_id != row["owner_id"]:
            raise RepositoryError(
                "Stored payload ownership does not match its database row."
            )

        if content_hash(payload.evidence) != row["evidence_hash"]:
            raise RepositoryError(
                "Stored evidence failed its integrity check."
            )

        if content_hash(payload) != row["report_hash"]:
            raise RepositoryError(
                "Stored report failed its integrity check."
            )

        self._validate_audit_trace(audit_trace)

        return SavedReport(
            report_id=row["report_id"],
            payload=payload,
            report_hash=row["report_hash"],
            approved_at=approved_at,
            audit_trace=audit_trace,
            idempotent_replay=idempotent_replay,
        )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Yield a short-lived row-aware SQLite connection."""

        connection = sqlite3.connect(
            self._database_path,
            timeout=5.0,
        )
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _models_json(models: Sequence[BaseModel]) -> str:
    """Serialize a sequence of Pydantic models deterministically."""

    return json.dumps(
        [model.model_dump(mode="json") for model in models],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def _required_identity(value: str) -> str:
    """Reject blank repository identifiers."""

    stripped = value.strip()
    if not stripped:
        raise ValueError("Repository identifier must not be blank.")
    return stripped
