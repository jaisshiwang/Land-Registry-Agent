"""Deterministic LangGraph orchestration for property research runs."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from typing import Literal, TypedDict, cast
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from pydantic import JsonValue, ValidationError

from land_registry_agent.analysis import (
    PropertyAnalyzer,
    content_hash,
)
from land_registry_agent.analysis import (
    verify_note as verify_note_claims,
)
from land_registry_agent.config import Settings
from land_registry_agent.data import DataSourceError, FetchResult, PropertyDataGateway
from land_registry_agent.intent import IntentFallback, interpret_request
from land_registry_agent.llm import LanguageServiceError, NoteDraftingService
from land_registry_agent.models import (
    ApprovalChoice,
    ApprovalDecision,
    AuditEvent,
    AuditStatus,
    ChartData,
    ChartSeries,
    ConfidenceLevel,
    EvidenceBundle,
    ExecutionPlan,
    HPIRecord,
    Intent,
    InterpretationMethod,
    MonthlyMetric,
    NoteDetailLevel,
    PersistenceMode,
    PlanStep,
    ReportPayload,
    SavedReport,
    SourceWindow,
    StepStatus,
    StreetMetric,
    ToolCategory,
    Transaction,
    TrendSummary,
    UserContext,
    VerificationResult,
)
from land_registry_agent.repository import ApprovedReportRepository, RepositoryError
from land_registry_agent.skills import compose_execution_plan


class WorkflowState(TypedDict, total=False):
    """Checkpointed state shared by LangGraph nodes."""

    run_id: str
    user_request: str
    user: UserContext
    report_name: str
    destination: str

    intent: Intent
    plan: ExecutionPlan

    transactions: tuple[Transaction, ...]
    price_paid_source_url: str
    price_paid_artifact_keys: tuple[str, ...]

    hpi_records: tuple[HPIRecord, ...]
    hpi_source_url: str | None
    hpi_artifact_keys: tuple[str, ...]

    evidence: EvidenceBundle
    evidence_hash: str

    draft: str
    draft_attempts: int
    verification: VerificationResult

    report_payload: ReportPayload
    report_hash: str
    approval: ApprovalDecision
    saved_report: SavedReport

    audit_trace: tuple[AuditEvent, ...]
    status: str
    error: str | None


ToolHandler = Callable[..., object]

CHECKPOINT_ALLOWED_TYPES = (
    UserContext,
    Intent,
    InterpretationMethod,
    PersistenceMode,
    ExecutionPlan,
    PlanStep,
    ToolCategory,
    StepStatus,
    Transaction,
    HPIRecord,
    EvidenceBundle,
    SourceWindow,
    MonthlyMetric,
    StreetMetric,
    TrendSummary,
    ChartData,
    ChartSeries,
    ConfidenceLevel,
    NoteDetailLevel,
    VerificationResult,
    ApprovalDecision,
    ApprovalChoice,
    ReportPayload,
    SavedReport,
    AuditEvent,
    AuditStatus,
)


@dataclass(frozen=True)
class ToolCapability:
    """A callable capability available to deterministic orchestration."""

    name: str
    category: ToolCategory
    handler: ToolHandler


class ToolRegistry:
    """Registry that categorises capabilities and gates writes."""

    def __init__(self, capabilities: tuple[ToolCapability, ...]) -> None:
        self._capabilities = {
            capability.name: capability for capability in capabilities
        }
        if len(self._capabilities) != len(capabilities):
            raise ValueError("Capability names must be unique.")

    @property
    def categories(self) -> Mapping[str, ToolCategory]:
        """Return safe capability metadata for diagnostics or the UI."""

        return {
            name: capability.category
            for name, capability in self._capabilities.items()
        }

    def invoke(
        self,
        name: str,
        *,
        approval_granted: bool = False,
        **kwargs: object,
    ) -> object:
        """Invoke a registered capability subject to the write gate."""

        try:
            capability = self._capabilities[name]
        except KeyError as exc:
            raise KeyError(f"Unknown capability: {name}") from exc

        if capability.category is ToolCategory.WRITE and not approval_granted:
            raise PermissionError(
                f"Write capability {name!r} requires explicit approval."
            )

        return capability.handler(**kwargs)


def build_execution_plan(run_id: str, intent: Intent) -> ExecutionPlan:
    """Compose a typed plan from deterministic execution skills."""

    return compose_execution_plan(run_id, intent)


def _update_plan_step(
    plan: ExecutionPlan,
    operation: str,
    status: StepStatus,
) -> ExecutionPlan:
    """Return a plan with one operation's status changed."""

    updated_steps = tuple(
        step.model_copy(update={"status": status})
        if step.operation == operation
        else step
        for step in plan.steps
    )
    return plan.model_copy(update={"steps": updated_steps})


def _update_plan_steps(
    plan: ExecutionPlan,
    operations: tuple[str, ...],
    status: StepStatus,
) -> ExecutionPlan:
    """Return a plan with several operation statuses changed."""

    for operation in operations:
        if any(step.operation == operation for step in plan.steps):
            plan = _update_plan_step(plan, operation, status)
    return plan


def _append_audit(
    state: WorkflowState,
    *,
    action: str,
    status: AuditStatus,
    explanation: str,
    metadata: dict[str, JsonValue] | None = None,
) -> tuple[AuditEvent, ...]:
    """Append a human-readable event without reasoning or secrets."""

    trace = state.get("audit_trace", ())
    event = AuditEvent(
        sequence=len(trace) + 1,
        action=action,
        status=status,
        explanation=explanation,
        metadata=metadata or {},
    )
    return (*trace, event)


class Orchestrator:
    """Coordinate the deterministic property-research workflow."""

    def __init__(
        self,
        *,
        settings: Settings,
        gateway: PropertyDataGateway,
        analyzer: PropertyAnalyzer,
        drafting_service: NoteDraftingService,
        repository: ApprovedReportRepository,
        intent_fallback: IntentFallback | None = None,
    ) -> None:
        self._settings = settings
        self._intent_fallback = intent_fallback
        self._repository = repository

        self._registry = ToolRegistry(
            (
                ToolCapability(
                    "fetch_price_paid_transactions",
                    ToolCategory.READ,
                    gateway.fetch_price_paid_transactions,
                ),
                ToolCapability(
                    "fetch_regional_hpi",
                    ToolCategory.READ,
                    gateway.fetch_regional_hpi,
                ),
                ToolCapability(
                    "analyse_local_trends",
                    ToolCategory.ANALYSIS,
                    analyzer.build_evidence,
                ),
                ToolCapability(
                    "compare_regional_hpi",
                    ToolCategory.ANALYSIS,
                    analyzer.build_evidence,
                ),
                ToolCapability(
                    "rank_high_value_streets",
                    ToolCategory.ANALYSIS,
                    analyzer.build_evidence,
                ),
                ToolCapability(
                    "build_chart_data",
                    ToolCategory.ANALYSIS,
                    analyzer.build_evidence,
                ),
                ToolCapability(
                    "draft_research_note",
                    ToolCategory.LANGUAGE,
                    drafting_service.draft,
                ),
                ToolCapability(
                    "verify_research_note",
                    ToolCategory.ANALYSIS,
                    verify_note_claims,
                ),
                ToolCapability(
                    "save_approved_report",
                    ToolCategory.WRITE,
                    repository.save_approved,
                ),
            )
        )

        settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_connection = sqlite3.connect(
            settings.checkpoint_db_path,
            check_same_thread=False,
        )
        serializer = JsonPlusSerializer(
            allowed_msgpack_modules=CHECKPOINT_ALLOWED_TYPES,
        )
        self._checkpointer = SqliteSaver(
            self._checkpoint_connection,
            serde=serializer,
        )
        self.graph = self._build_graph()

    @property
    def capability_categories(self) -> Mapping[str, ToolCategory]:
        """Expose capability names and categories without their handlers."""

        return self._registry.categories

    def initialise(self) -> None:
        """Initialise persistent report storage explicitly."""

        self._repository.initialize()

    def close(self) -> None:
        """Close the SQLite checkpoint connection."""

        self._checkpoint_connection.close()

    def prepare(
        self,
        *,
        user_request: str,
        user: UserContext,
        report_name: str = "",
        destination: str = "SQLite tracking sheet",
        run_id: str | None = None,
    ) -> WorkflowState:
        """Interpret the request and stop after planning, before data reads."""

        resolved_run_id = run_id or f"run_{uuid4().hex}"
        initial_state: WorkflowState = {
            "run_id": resolved_run_id,
            "user_request": user_request,
            "user": user,
            "report_name": report_name.strip(),
            "destination": destination,
            "draft_attempts": 0,
            "audit_trace": (),
            "status": "new",
            "error": None,
        }
        result = self.graph.invoke(
            initial_state,
            config=self._config(resolved_run_id),
            interrupt_after=["create_plan"],
        )
        return cast(WorkflowState, result)

    def execute(self, run_id: str) -> WorkflowState:
        """Continue a planned run until completion or approval interruption."""

        result = self.graph.invoke(None, config=self._config(run_id))
        return cast(WorkflowState, result)

    def resume_approval(
        self,
        *,
        run_id: str,
        decision: ApprovalDecision,
    ) -> WorkflowState:
        """Resume an approval interrupt with a typed decision."""

        result = self.graph.invoke(
            Command(resume=decision.model_dump(mode="json")),
            config=self._config(run_id),
        )
        return cast(WorkflowState, result)

    def get_state(self, run_id: str) -> WorkflowState:
        """Return the latest checkpointed state for a run."""

        snapshot = self.graph.get_state(self._config(run_id))
        return cast(WorkflowState, snapshot.values)

    @staticmethod
    def _config(run_id: str) -> RunnableConfig:
        return {"configurable": {"thread_id": run_id}}

    def _build_graph(
        self,
    ) -> CompiledStateGraph[WorkflowState, None, WorkflowState, WorkflowState]:
        builder = StateGraph(WorkflowState)

        builder.add_node("initialise_run", self._initialise_run)
        builder.add_node("interpret_request", self._interpret_request)
        builder.add_node("create_plan", self._create_plan)
        builder.add_node("fetch_data", self._fetch_data)
        builder.add_node("analyse_data", self._analyse_data)
        builder.add_node("draft_note", self._draft_note)
        builder.add_node("verify_note", self._verify_note)
        builder.add_node("prepare_report", self._prepare_report)
        builder.add_node("request_approval", self._request_approval)
        builder.add_node("save_report", self._save_report)
        builder.add_node("request_clarification", self._request_clarification)
        builder.add_node("complete", self._complete)
        builder.add_node("fail", self._fail)

        builder.add_edge(START, "initialise_run")
        builder.add_edge("initialise_run", "interpret_request")
        builder.add_conditional_edges(
            "interpret_request",
            self._route_after_interpretation,
            {
                "plan": "create_plan",
                "clarify": "request_clarification",
                "fail": "fail",
            },
        )
        builder.add_edge("create_plan", "fetch_data")
        builder.add_conditional_edges(
            "fetch_data",
            self._route_after_operation,
            {"continue": "analyse_data", "fail": "fail"},
        )
        builder.add_conditional_edges(
            "analyse_data",
            self._route_after_analysis,
            {"draft": "draft_note", "complete": "complete", "fail": "fail"},
        )
        builder.add_conditional_edges(
            "draft_note",
            self._route_after_operation,
            {"continue": "verify_note", "fail": "fail"},
        )
        builder.add_conditional_edges(
            "verify_note",
            self._route_after_verification,
            {
                "redraft": "draft_note",
                "prepare_report": "prepare_report",
                "complete": "complete",
                "fail": "fail",
            },
        )
        builder.add_edge("prepare_report", "request_approval")
        builder.add_conditional_edges(
            "request_approval",
            self._route_after_approval,
            {"save": "save_report", "complete": "complete", "fail": "fail"},
        )
        builder.add_conditional_edges(
            "save_report",
            self._route_after_operation,
            {"continue": "complete", "fail": "fail"},
        )
        builder.add_edge("request_clarification", END)
        builder.add_edge("complete", END)
        builder.add_edge("fail", END)

        return builder.compile(checkpointer=self._checkpointer)

    def _initialise_run(self, state: WorkflowState) -> WorkflowState:
        return {
            "status": "interpreting",
            "audit_trace": _append_audit(
                state,
                action="initialise_run",
                status=AuditStatus.SUCCEEDED,
                explanation=(
                    f"Started research run {state['run_id']} for "
                    f"{state['user'].display_name}."
                ),
                metadata={
                    "run_id": state["run_id"],
                    "owner_id": state["user"].user_id,
                },
            ),
        }

    def _interpret_request(self, state: WorkflowState) -> WorkflowState:
        try:
            intent = interpret_request(
                state["user_request"],
                fallback=self._intent_fallback,
            )
        except (ValueError, LanguageServiceError) as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "audit_trace": _append_audit(
                    state,
                    action="interpret_request",
                    status=AuditStatus.FAILED,
                    explanation="The request could not be interpreted safely.",
                ),
            }

        explanation = (
            f"Validated request for {intent.postcode} over "
            f"{intent.requested_years} years."
            if not intent.clarification_reason
            else "The request requires clarification before data retrieval."
        )
        return {
            "intent": intent,
            "status": (
                "clarification_required"
                if intent.clarification_reason
                else "planning"
            ),
            "audit_trace": _append_audit(
                state,
                action="interpret_request",
                status=(
                    AuditStatus.PAUSED
                    if intent.clarification_reason
                    else AuditStatus.SUCCEEDED
                ),
                explanation=explanation,
                metadata={
                    "interpretation_method": intent.interpretation_method.value,
                    "confidence": intent.confidence,
                },
            ),
        }

    def _create_plan(self, state: WorkflowState) -> WorkflowState:
        plan = build_execution_plan(state["run_id"], state["intent"])
        return {
            "plan": plan,
            "status": "planned",
            "audit_trace": _append_audit(
                state,
                action="create_plan",
                status=AuditStatus.SUCCEEDED,
                explanation=(
                    f"Composed {len(plan.selected_skills)} deterministic "
                    f"execution skills into {len(plan.steps)} plan steps "
                    "before external data retrieval."
                ),
                metadata={
                    "selected_skills": list(plan.selected_skills),
                    "step_count": len(plan.steps),
                },
            ),
        }

    def _fetch_data(self, state: WorkflowState) -> WorkflowState:
        plan = _update_plan_step(
            state["plan"],
            "fetch_price_paid_transactions",
            StepStatus.IN_PROGRESS,
        )
        try:
            local_result = cast(
                FetchResult[Transaction],
                self._registry.invoke(
                    "fetch_price_paid_transactions",
                    postcode=cast(str, state["intent"].postcode),
                    requested_years=state["intent"].requested_years,
                ),
            )
            plan = _update_plan_step(
                plan,
                "fetch_price_paid_transactions",
                StepStatus.COMPLETED,
            )

            hpi_records: tuple[HPIRecord, ...] = ()
            hpi_source_url: str | None = None
            hpi_artifact_keys: tuple[str, ...] = ()
            if state["intent"].regional_comparison_requested:
                plan = _update_plan_step(
                    plan,
                    "fetch_regional_hpi",
                    StepStatus.IN_PROGRESS,
                )
                regional_result = cast(
                    FetchResult[HPIRecord],
                    self._registry.invoke(
                        "fetch_regional_hpi",
                        region=cast(str, state["intent"].region),
                        requested_years=state["intent"].requested_years,
                    ),
                )
                hpi_records = regional_result.records
                hpi_source_url = regional_result.source_url
                hpi_artifact_keys = regional_result.artifact_keys
                plan = _update_plan_step(
                    plan,
                    "fetch_regional_hpi",
                    StepStatus.COMPLETED,
                )
        except (DataSourceError, ValueError, KeyError) as exc:
            return {
                "plan": _update_plan_steps(
                    plan,
                    ("fetch_price_paid_transactions", "fetch_regional_hpi"),
                    StepStatus.FAILED,
                ),
                "status": "failed",
                "error": str(exc),
                "audit_trace": _append_audit(
                    state,
                    action="fetch_data",
                    status=AuditStatus.FAILED,
                    explanation="An external data source failed after bounded retries.",
                ),
            }

        return {
            "transactions": local_result.records,
            "price_paid_source_url": local_result.source_url,
            "price_paid_artifact_keys": local_result.artifact_keys,
            "hpi_records": hpi_records,
            "hpi_source_url": hpi_source_url,
            "hpi_artifact_keys": hpi_artifact_keys,
            "plan": plan,
            "status": "analysing",
            "audit_trace": _append_audit(
                state,
                action="fetch_data",
                status=AuditStatus.SUCCEEDED,
                explanation=(
                    f"Retrieved {len(local_result.records)} Price Paid transactions"
                    + (
                        f" and {len(hpi_records)} regional HPI records."
                        if state["intent"].regional_comparison_requested
                        else "; regional data was not requested."
                    )
                ),
                metadata={
                    "transaction_count": len(local_result.records),
                    "hpi_record_count": len(hpi_records),
                },
            ),
        }

    def _analyse_data(self, state: WorkflowState) -> WorkflowState:
        operations = (
            "analyse_local_trends",
            "compare_regional_hpi",
            "rank_high_value_streets",
            "build_chart_data",
        )
        plan = _update_plan_steps(
            state["plan"], operations, StepStatus.IN_PROGRESS
        )
        try:
            evidence = cast(
                EvidenceBundle,
                self._registry.invoke(
                    "analyse_local_trends",
                    user_request=state["user_request"],
                    intent=state["intent"],
                    transactions=state["transactions"],
                    hpi_records=state.get("hpi_records", ()),
                    price_paid_source_url=state["price_paid_source_url"],
                    price_paid_artifact_keys=state.get(
                        "price_paid_artifact_keys", ()
                    ),
                    hpi_source_url=state.get("hpi_source_url"),
                    hpi_artifact_keys=state.get("hpi_artifact_keys", ()),
                ),
            )
        except ValueError as exc:
            return {
                "plan": _update_plan_steps(plan, operations, StepStatus.FAILED),
                "status": "failed",
                "error": str(exc),
                "audit_trace": _append_audit(
                    state,
                    action="analyse_data",
                    status=AuditStatus.FAILED,
                    explanation="Deterministic analysis could not be completed.",
                ),
            }

        return {
            "evidence": evidence,
            "evidence_hash": content_hash(evidence),
            "plan": _update_plan_steps(plan, operations, StepStatus.COMPLETED),
            "status": "drafting",
            "audit_trace": _append_audit(
                state,
                action="analyse_data",
                status=AuditStatus.SUCCEEDED,
                explanation=(
                    "Calculated local metrics, applicable comparisons, street "
                    "rankings and chart-ready data."
                ),
                metadata={
                    "confidence": evidence.confidence.value,
                    "limitation_count": len(evidence.limitations),
                },
            ),
        }

    def _draft_note(self, state: WorkflowState) -> WorkflowState:
        attempts = state.get("draft_attempts", 0) + 1
        previous_verification = state.get("verification")
        unsupported_claims = (
            previous_verification.unsupported_claims
            if previous_verification is not None
            else ()
        )
        try:
            draft = cast(
                str,
                self._registry.invoke(
                    "draft_research_note",
                    evidence=state["evidence"],
                    unsupported_claims=unsupported_claims,
                ),
            )
        except LanguageServiceError as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "plan": _update_plan_step(
                    state["plan"],
                    "draft_research_note",
                    StepStatus.FAILED,
                ),
                "audit_trace": _append_audit(
                    state,
                    action="draft_research_note",
                    status=AuditStatus.FAILED,
                    explanation="The research note could not be drafted.",
                ),
            }

        return {
            "draft": draft,
            "draft_attempts": attempts,
            "plan": _update_plan_step(
                state["plan"],
                "draft_research_note",
                StepStatus.COMPLETED,
            ),
            "status": "verifying",
            "audit_trace": _append_audit(
                state,
                action="draft_research_note",
                status=AuditStatus.SUCCEEDED,
                explanation=(
                    "Drafted the research note from the frozen evidence bundle "
                    f"(attempt {attempts})."
                ),
                metadata={"attempt": attempts},
            ),
        }

    def _verify_note(self, state: WorkflowState) -> WorkflowState:
        verification = cast(
            VerificationResult,
            self._registry.invoke(
                "verify_research_note",
                note=state["draft"],
                evidence=state["evidence"],
            ),
        )
        plan = state["plan"]
        if verification.supported:
            plan = _update_plan_step(
                plan, "verify_research_note", StepStatus.COMPLETED
            )
        elif state["draft_attempts"] > self._settings.maximum_corrective_redrafts:
            plan = _update_plan_step(
                plan, "verify_research_note", StepStatus.FAILED
            )

        can_redraft = (
            state["draft_attempts"]
            <= self._settings.maximum_corrective_redrafts
        )
        return {
            "verification": verification,
            "plan": plan,
            "status": "verified" if verification.supported else "verification_failed",
            "error": (
                None
                if verification.supported or can_redraft
                else "The note still contains unsupported claims."
            ),
            "audit_trace": _append_audit(
                state,
                action="verify_research_note",
                status=(
                    AuditStatus.SUCCEEDED
                    if verification.supported
                    else AuditStatus.FAILED
                ),
                explanation=(
                    "Verified all numerical and date claims against evidence."
                    if verification.supported
                    else (
                        "Found unsupported claims; one corrective redraft will be "
                        "attempted."
                        if can_redraft
                        else "Verification failed closed after the redraft."
                    )
                ),
                metadata={
                    "unsupported_claim_count": len(
                        verification.unsupported_claims
                    )
                },
            ),
        }

    def _prepare_report(self, state: WorkflowState) -> WorkflowState:
        intent = state["intent"]
        evidence = state["evidence"]
        report_name = state.get("report_name", "").strip()
        if not report_name:
            report_name = (
                f"{intent.postcode} Property Trends - {intent.requested_years} "
                f"Years - {date.today().isoformat()}"
            )

        payload = ReportPayload(
            run_id=state["run_id"],
            owner=state["user"],
            report_name=report_name,
            destination=state["destination"],
            user_request=state["user_request"],
            postcode=cast(str, intent.postcode),
            region=intent.region,
            requested_years=intent.requested_years,
            research_note=state["draft"],
            charts=evidence.charts,
            evidence=evidence,
            evidence_hash=state["evidence_hash"],
        )
        report_hash = content_hash(payload)
        return {
            "report_payload": payload,
            "report_hash": report_hash,
            "status": "awaiting_approval",
            "plan": _update_plan_step(
                state["plan"], "request_approval", StepStatus.IN_PROGRESS
            ),
            "audit_trace": _append_audit(
                state,
                action="request_approval",
                status=AuditStatus.PAUSED,
                explanation=(
                    "Paused before saving because persistence requires explicit "
                    "approval of the exact report payload."
                ),
                metadata={"report_hash": report_hash},
            ),
        }

    def _request_approval(self, state: WorkflowState) -> WorkflowState:
        # This node deliberately performs no work before the interrupt.
        raw_decision = interrupt(
            {
                "run_id": state["run_id"],
                "owner": state["user"].model_dump(mode="json"),
                "destination": state["destination"],
                "report_name": state["report_payload"].report_name,
                "report_hash": state["report_hash"],
                "report": state["report_payload"].model_dump(mode="json"),
                "instruction": "Approve or reject this exact frozen report payload.",
            }
        )
        try:
            decision = ApprovalDecision.model_validate(raw_decision)
        except ValidationError as exc:
            return {
                "status": "failed",
                "error": f"Invalid approval decision: {exc}",
                "plan": _update_plan_step(
                    state["plan"], "request_approval", StepStatus.FAILED
                ),
                "audit_trace": _append_audit(
                    state,
                    action="request_approval",
                    status=AuditStatus.FAILED,
                    explanation="The approval response was invalid.",
                ),
            }

        mismatch = (
            decision.run_id != state["run_id"]
            or decision.owner_id != state["user"].user_id
            or decision.report_hash != state["report_hash"]
        )
        if mismatch:
            return {
                "approval": decision,
                "status": "failed",
                "error": "Approval refers to a stale or different report.",
                "plan": _update_plan_step(
                    state["plan"], "request_approval", StepStatus.FAILED
                ),
                "audit_trace": _append_audit(
                    state,
                    action="request_approval",
                    status=AuditStatus.FAILED,
                    explanation="Rejected a stale or mismatched approval payload.",
                ),
            }

        if decision.choice is ApprovalChoice.REJECT:
            return {
                "approval": decision,
                "status": "rejected",
                "plan": _update_plan_step(
                    _update_plan_step(
                        state["plan"],
                        "request_approval",
                        StepStatus.COMPLETED,
                    ),
                    "save_approved_report",
                    StepStatus.SKIPPED,
                ),
                "audit_trace": _append_audit(
                    state,
                    action="request_approval",
                    status=AuditStatus.REJECTED,
                    explanation="The user rejected the proposed report.",
                ),
            }

        return {
            "approval": decision,
            "status": "approved",
            "plan": _update_plan_step(
                state["plan"], "request_approval", StepStatus.COMPLETED
            ),
            "audit_trace": _append_audit(
                state,
                action="request_approval",
                status=AuditStatus.SUCCEEDED,
                explanation="The user approved the exact report hash for saving.",
                metadata={"report_hash": decision.report_hash},
            ),
        }

    def _save_report(self, state: WorkflowState) -> WorkflowState:
        try:
            saved = cast(
                SavedReport,
                self._registry.invoke(
                    "save_approved_report",
                    approval_granted=True,
                    payload=state["report_payload"],
                    approval=state["approval"],
                    audit_trace=state["audit_trace"],
                ),
            )
        except (RepositoryError, PermissionError, ValueError) as exc:
            return {
                "status": "failed",
                "error": str(exc),
                "plan": _update_plan_step(
                    state["plan"], "save_approved_report", StepStatus.FAILED
                ),
                "audit_trace": _append_audit(
                    state,
                    action="save_approved_report",
                    status=AuditStatus.FAILED,
                    explanation="The approved report could not be saved.",
                ),
            }

        return {
            "saved_report": saved,
            "audit_trace": saved.audit_trace,
            "plan": _update_plan_step(
                state["plan"], "save_approved_report", StepStatus.COMPLETED
            ),
            "status": "saved",
        }

    def _request_clarification(self, state: WorkflowState) -> WorkflowState:
        return {
            "status": "clarification_required",
            "error": state["intent"].clarification_reason,
        }

    def _complete(self, state: WorkflowState) -> WorkflowState:
        rejected = state.get("status") == "rejected"
        return {
            "status": "rejected" if rejected else "completed",
            "audit_trace": _append_audit(
                state,
                action="complete",
                status=(
                    AuditStatus.REJECTED if rejected else AuditStatus.SUCCEEDED
                ),
                explanation=(
                    "Completed the run without saving after rejection."
                    if rejected
                    else "Completed the requested property research workflow."
                ),
            ),
        }

    @staticmethod
    def _fail(state: WorkflowState) -> WorkflowState:
        return {
            "status": "failed",
            "error": state.get("error") or "The workflow failed safely.",
        }

    @staticmethod
    def _route_after_interpretation(
        state: WorkflowState,
    ) -> Literal["plan", "clarify", "fail"]:
        if state.get("error"):
            return "fail"
        if state["intent"].clarification_reason:
            return "clarify"
        return "plan"

    @staticmethod
    def _route_after_operation(
        state: WorkflowState,
    ) -> Literal["continue", "fail"]:
        return "fail" if state.get("error") else "continue"

    @staticmethod
    def _route_after_analysis(
        state: WorkflowState,
    ) -> Literal["draft", "complete", "fail"]:
        if state.get("error"):
            return "fail"
        return "draft" if state["intent"].note_requested else "complete"

    def _route_after_verification(
        self,
        state: WorkflowState,
    ) -> Literal["redraft", "prepare_report", "complete", "fail"]:
        if state["verification"].supported:
            if state["intent"].persistence_mode is PersistenceMode.REQUESTED:
                return "prepare_report"
            return "complete"
        if (
            state["draft_attempts"]
            <= self._settings.maximum_corrective_redrafts
        ):
            return "redraft"
        return "fail"

    @staticmethod
    def _route_after_approval(
        state: WorkflowState,
    ) -> Literal["save", "complete", "fail"]:
        if state.get("error"):
            return "fail"
        if state["approval"].choice is ApprovalChoice.REJECT:
            return "complete"
        return "save"
