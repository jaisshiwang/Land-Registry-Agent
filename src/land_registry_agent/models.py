"""Core typed contracts for the property research workflow."""

from __future__ import annotations

from datetime import UTC, date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    PositiveFloat,
    PositiveInt,
    field_validator,
    model_validator,
)


def utc_now() -> datetime:
    """Return the current time as a timezone-aware UTC value."""

    return datetime.now(UTC)


Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class DomainModel(BaseModel):
    """Base for immutable, strict domain contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InterpretationMethod(StrEnum):
    DETERMINISTIC = "deterministic"
    LLM_ASSISTED = "llm_assisted"


class PersistenceMode(StrEnum):
    REQUESTED = "requested"
    NOT_REQUESTED = "not_requested"
    FORBIDDEN = "forbidden"


class ToolCategory(StrEnum):
    READ = "read"
    ANALYSIS = "analysis"
    LANGUAGE = "language"
    WRITE = "write"


class StepStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class ConfidenceLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class NoteDetailLevel(StrEnum):
    CONCISE = "concise"
    DETAILED = "detailed"


class AuditStatus(StrEnum):
    STARTED = "started"
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"
    FAILED = "failed"
    PAUSED = "paused"
    REJECTED = "rejected"


class ApprovalChoice(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class UserContext(DomainModel):
    """Demonstration identity used for application-level ownership."""

    user_id: str = Field(min_length=1, max_length=100)
    display_name: str = Field(min_length=1, max_length=200)

    @field_validator("user_id", "display_name")
    @classmethod
    def reject_blank_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Identity values must not be blank.")
        return value


class Intent(DomainModel):
    """Natural-language request interpreted into supported workflow options."""

    postcode: str | None = None
    requested_years: int = Field(default=3, ge=1, le=20)
    region: str | None = None
    regional_comparison_requested: bool = False
    street_ranking_requested: bool = False
    note_requested: bool = True
    note_detail_level: NoteDetailLevel = NoteDetailLevel.DETAILED
    note_paragraph_count: int | None = Field(default=None, ge=1, le=5)
    persistence_mode: PersistenceMode = PersistenceMode.NOT_REQUESTED
    latest_available_data: bool = True
    live_refresh_requested: bool = False
    interpretation_method: InterpretationMethod
    confidence: float = Field(ge=0.0, le=1.0)
    clarification_reason: str | None = None

    @field_validator("postcode")
    @classmethod
    def normalise_postcode(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalised = " ".join(value.upper().split())
        return normalised or None

    @field_validator("region")
    @classmethod
    def normalise_region(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class PlanStep(DomainModel):
    """One request-specific operation visible to the user."""

    sequence: PositiveInt
    operation: str = Field(min_length=1, max_length=100)
    category: ToolCategory
    description: str = Field(min_length=1, max_length=300)
    status: StepStatus = StepStatus.PENDING


class ExecutionPlan(DomainModel):
    """Plan created before any registered external read executes."""

    run_id: str = Field(min_length=1)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    selected_skills: tuple[str, ...] = ()
    steps: tuple[PlanStep, ...]

    @model_validator(mode="after")
    def require_valid_plan(self) -> Self:
        expected = list(range(1, len(self.steps) + 1))
        actual = [step.sequence for step in self.steps]
        if actual != expected:
            raise ValueError("Plan step sequences must be contiguous and ordered.")

        if any(not skill.strip() for skill in self.selected_skills):
            raise ValueError("Selected skill names must not be blank.")

        if len(set(self.selected_skills)) != len(self.selected_skills):
            raise ValueError("Selected skill names must be unique.")

        return self


PropertyType = Literal[
    "detached",
    "semi_detached",
    "terraced",
    "flat_maisonette",
    "other",
    "unknown",
]


class Transaction(DomainModel):
    """Normalised flat Price Paid transaction."""

    transaction_id: str = Field(min_length=1)
    transfer_date: date
    price_gbp: PositiveInt
    postcode: str = Field(min_length=2, max_length=10)
    property_type: PropertyType
    street: str | None = None


class HPIRecord(DomainModel):
    """Normalised regional House Price Index observation."""

    period: date
    region: str = Field(min_length=1)
    average_price_gbp: PositiveFloat
    annual_change_percentage: float | None = None
    monthly_change_percentage: float | None = None


class SourceWindow(DomainModel):
    """Date window actually used from an external source."""

    source_name: str = Field(min_length=1)
    start_date: date
    end_date: date
    latest_available_date: date

    @model_validator(mode="after")
    def require_valid_dates(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError("Source end date precedes its start date.")
        if self.latest_available_date < self.end_date:
            raise ValueError("Latest available date precedes the used end date.")
        return self


class MonthlyMetric(DomainModel):
    """Locally calculated monthly property-price values."""

    period: date
    transaction_count: PositiveInt
    median_price_gbp: PositiveFloat
    mean_price_gbp: PositiveFloat


class StreetMetric(DomainModel):
    """Qualifying street aggregation after applying the sample-size rule."""

    rank: PositiveInt
    street: str = Field(min_length=1)
    median_price_gbp: PositiveFloat
    transaction_count: int = Field(ge=3)


class TrendSummary(DomainModel):
    """Start, end, and percentage-change evidence for one trend."""

    label: str = Field(min_length=1)
    start_value: float | None = None
    end_value: float | None = None
    percentage_change: float | None = None
    change_claim_permitted: bool


class ChartSeries(DomainModel):
    """One immutable numeric series in a chart."""

    name: str = Field(min_length=1)
    values: tuple[float, ...]
    sample_sizes: tuple[int, ...] | None = None

    @model_validator(mode="after")
    def require_matching_sample_sizes(self) -> Self:
        if (
            self.sample_sizes is not None
            and len(self.sample_sizes) != len(self.values)
        ):
            raise ValueError("Chart sample sizes must match the series length.")
        return self


class ChartData(DomainModel):
    """Renderer-independent chart data stored with the report."""

    chart_id: str = Field(min_length=1)
    chart_type: Literal["line", "bar"]
    title: str = Field(min_length=1)
    labels: tuple[str, ...]
    series: tuple[ChartSeries, ...]

    @model_validator(mode="after")
    def require_matching_series_lengths(self) -> Self:
        if any(len(series.values) != len(self.labels) for series in self.series):
            raise ValueError("Every chart series must match the label count.")
        return self


class EvidenceBundle(DomainModel):
    """Frozen evidence supplied to drafting and verification."""

    user_request: str = Field(min_length=1)
    intent: Intent
    source_windows: tuple[SourceWindow, ...]
    monthly_local_metrics: tuple[MonthlyMetric, ...]
    local_trend: TrendSummary
    regional_trend: TrendSummary | None = None
    periods_overlap: bool | None = None
    regional_comparison_claim_permitted: bool = False
    comparison_difference_percentage_points: float | None = None
    street_rankings: tuple[StreetMetric, ...] = ()
    confidence: ConfidenceLevel
    limitations: tuple[str, ...] = ()
    charts: tuple[ChartData, ...] = ()
    source_urls: tuple[str, ...] = ()
    artifact_keys: tuple[str, ...] = ()


class VerificationResult(DomainModel):
    """Deterministic result of checking a drafted note."""

    supported: bool
    checked_claim_count: int = Field(ge=0)
    unsupported_claims: tuple[str, ...] = ()


class AuditEvent(DomainModel):
    """Human-readable event without hidden reasoning or secrets."""

    sequence: PositiveInt
    timestamp: AwareDatetime = Field(default_factory=utc_now)
    action: str = Field(min_length=1, max_length=100)
    status: AuditStatus
    explanation: str = Field(min_length=1, max_length=1_000)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ReportPayload(DomainModel):
    """Exact report content frozen and hashed before approval."""

    run_id: str = Field(min_length=1)
    owner: UserContext
    report_name: str = Field(min_length=1, max_length=200)
    destination: str = Field(min_length=1, max_length=200)
    created_at: AwareDatetime = Field(default_factory=utc_now)
    user_request: str = Field(min_length=1)
    postcode: str = Field(min_length=2, max_length=10)
    region: str | None = None
    requested_years: int = Field(ge=1, le=20)
    research_note: str = Field(min_length=1, max_length=2_000)
    charts: tuple[ChartData, ...]
    evidence: EvidenceBundle
    evidence_hash: Sha256Hex


class ApprovalDecision(DomainModel):
    """Explicit decision supplied when the paused graph resumes."""

    run_id: str = Field(min_length=1)
    owner_id: str = Field(min_length=1)
    report_hash: Sha256Hex
    choice: ApprovalChoice
    decided_at: AwareDatetime = Field(default_factory=utc_now)


class SavedReport(DomainModel):
    """Approved report returned by the owner-scoped repository."""

    report_id: str = Field(pattern=r"^rpt_[A-Za-z0-9_-]+$")
    payload: ReportPayload
    report_hash: Sha256Hex
    approved_at: AwareDatetime
    audit_trace: tuple[AuditEvent, ...]
    idempotent_replay: bool = False


class ModelDiagnosticResult(DomainModel):
    """Result from the optional configured-model access check."""

    service_name: str = Field(min_length=1, max_length=100)
    model_name: str = Field(min_length=1, max_length=200)
    accessible: bool
    explanation: str = Field(min_length=1, max_length=500)
