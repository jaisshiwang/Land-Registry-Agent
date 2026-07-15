"""Streamlit demonstration UI for the property research orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from land_registry_agent.analysis import PropertyAnalyzer
from land_registry_agent.config import Settings
from land_registry_agent.data import LandRegistryGateway
from land_registry_agent.llm import (
    OpenAIIntentService,
    OpenAINoteDraftingService,
)
from land_registry_agent.models import (
    ApprovalChoice,
    ApprovalDecision,
    ChartData,
    EvidenceBundle,
    InterpretationMethod,
    NoteDetailLevel,
    SavedReport,
    UserContext,
)
from land_registry_agent.orchestrator import Orchestrator, WorkflowState
from land_registry_agent.repository import SQLiteReportRepository

EXAMPLE_REQUEST = (
    "Analyse property price trends in GU1 over the last 3 years. "
    "Compare with the South East regional average. Identify the "
    "highest-value streets. Then prepare a one-paragraph research note "
    "and add it to my tracking sheet."
)


@dataclass(frozen=True)
class AppServices:
    """Long-lived application services cached by Streamlit."""

    settings: Settings
    repository: SQLiteReportRepository
    orchestrator: Orchestrator | None


@st.cache_resource
def create_services() -> AppServices:
    """Construct application services once per Streamlit server process."""

    settings = Settings()
    repository = SQLiteReportRepository(settings.reports_db_path)

    if not settings.openai_enabled:
        repository.initialize()
        return AppServices(
            settings=settings,
            repository=repository,
            orchestrator=None,
        )

    gateway = LandRegistryGateway(settings)
    orchestrator = Orchestrator(
        settings=settings,
        gateway=gateway,
        analyzer=PropertyAnalyzer(settings),
        drafting_service=OpenAINoteDraftingService(settings),
        repository=repository,
        intent_fallback=OpenAIIntentService(settings),
    )
    orchestrator.initialise()

    return AppServices(
        settings=settings,
        repository=repository,
        orchestrator=orchestrator,
    )


def render_demo_user(settings: Settings) -> UserContext:
    """Render the demonstration identity selector."""

    st.sidebar.header("Demonstration identity")
    st.sidebar.caption(
        "This demonstrates owner-scoped records, not secure authentication."
    )

    user_id = st.sidebar.text_input(
        "User ID",
        value=settings.demo_user_id,
    ).strip()
    display_name = st.sidebar.text_input(
        "Display name",
        value=settings.demo_user_display_name,
    ).strip()

    try:
        return UserContext(
            user_id=user_id,
            display_name=display_name,
        )
    except ValidationError as exc:
        st.sidebar.error(f"Invalid demonstration identity: {exc}")
        st.stop()


def render_request_form(
    orchestrator: Orchestrator | None,
    user: UserContext,
) -> WorkflowState | None:
    """Render request inputs and prepare a plan without external reads."""

    with st.form("research-request", clear_on_submit=False):
        request = st.text_area(
            "Property research request",
            value=EXAMPLE_REQUEST,
            height=150,
        )
        report_name = st.text_input(
            "Report name",
            placeholder=(
                "Optional — a default is generated if persistence is requested"
            ),
        )
        submitted = st.form_submit_button(
            "Create execution plan",
            type="primary",
            disabled=orchestrator is None,
        )

    if not submitted or orchestrator is None:
        return cast(
            WorkflowState | None,
            st.session_state.get("workflow_state"),
        )

    try:
        state = orchestrator.prepare(
            user_request=request,
            user=user,
            report_name=report_name,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not prepare the workflow: {exc}")
        return None

    st.session_state["workflow_state"] = state
    return state


def render_intent_summary(state: WorkflowState) -> None:
    """Explain how the request was interpreted."""

    intent = state.get("intent")
    if intent is None:
        return

    st.subheader("Request interpretation")

    method = (
        "LLM-assisted"
        if intent.interpretation_method is InterpretationMethod.LLM_ASSISTED
        else "Deterministic"
    )
    st.write(f"**Method:** {method}")
    st.write(f"**Confidence:** {intent.confidence:.0%}")

    if intent.interpretation_method is InterpretationMethod.LLM_ASSISTED:
        st.info(
            "LLM assistance was used because deterministic parsing found "
            "unresolved or ambiguous wording. The model only converted the "
            "request into typed intent fields; it did not call data tools, "
            "perform calculations, approve a write, or save anything."
        )
    else:
        st.success(
            "The request was fully recognised by deterministic parsing. "
            "No intent-model call was required."
        )

    if not intent.note_requested:
        paragraph_display = "Not requested"
    elif intent.note_paragraph_count is not None:
        paragraph_display = str(intent.note_paragraph_count)
    elif intent.note_detail_level is NoteDetailLevel.CONCISE:
        paragraph_display = "1 (concise default)"
    else:
        paragraph_display = "2–4 (detailed default)"

    st.dataframe(
        [
            {
                "Field": "Postcode",
                "Interpreted value": intent.postcode or "—",
            },
            {
                "Field": "Requested years",
                "Interpreted value": str(intent.requested_years),
            },
            {
                "Field": "Regional comparison",
                "Interpreted value": (
                    "Yes" if intent.regional_comparison_requested else "No"
                ),
            },
            {"Field": "Region", "Interpreted value": intent.region or "—"},
            {
                "Field": "Street ranking",
                "Interpreted value": (
                    "Yes" if intent.street_ranking_requested else "No"
                ),
            },
            {
                "Field": "Research note",
                "Interpreted value": (
                    "Yes" if intent.note_requested else "No"
                ),
            },
            {
                "Field": "Note detail",
                "Interpreted value": (
                    intent.note_detail_level.value.title()
                    if intent.note_requested
                    else "—"
                ),
            },
            {
                "Field": "Note paragraphs",
                "Interpreted value": paragraph_display,
            },
            {
                "Field": "Persistence",
                "Interpreted value": intent.persistence_mode.value,
            },
        ],
        hide_index=True,
        width="stretch",
    )


def render_plan(state: WorkflowState) -> None:
    """Display the request-specific plan and its current statuses."""

    plan = state.get("plan")
    if plan is None:
        return

    st.subheader("Execution plan")
    if plan.selected_skills:
        skill_labels = ", ".join(
            f"`{skill}`" for skill in plan.selected_skills
        )
        st.markdown(f"**Selected deterministic skills:** {skill_labels}")

    st.dataframe(
        [
            {
                "Step": step.sequence,
                "Operation": step.operation,
                "Category": step.category.value,
                "Status": step.status.value,
                "Purpose": step.description,
            }
            for step in plan.steps
        ],
        hide_index=True,
        width="stretch",
    )


def render_evidence(evidence: EvidenceBundle) -> None:
    """Display compact deterministic evidence and chart-ready data."""

    st.subheader("Evidence summary")

    transaction_count = sum(
        metric.transaction_count for metric in evidence.monthly_local_metrics
    )
    local = evidence.local_trend

    columns = st.columns(4)
    columns[0].metric("Transactions", transaction_count)
    columns[1].metric("Confidence", evidence.confidence.value.title())
    columns[2].metric("Local start", format_pounds(local.start_value))
    columns[3].metric(
        "Local change",
        format_percentage(local.percentage_change),
    )

    if evidence.regional_trend is not None:
        regional = evidence.regional_trend
        regional_columns = st.columns(3)
        regional_columns[0].metric(
            "Regional start",
            format_pounds(regional.start_value),
        )
        regional_columns[1].metric(
            "Regional end",
            format_pounds(regional.end_value),
        )
        regional_columns[2].metric(
            "Regional change",
            format_percentage(regional.percentage_change),
        )

        if evidence.periods_overlap is False:
            st.warning(
                "The local and regional source periods do not overlap. "
                "A like-for-like performance claim is prohibited."
            )

    if evidence.source_windows:
        st.caption("Source windows")
        st.dataframe(
            [
                {
                    "Source": window.source_name,
                    "Used from": window.start_date.isoformat(),
                    "Used to": window.end_date.isoformat(),
                    "Latest available": window.latest_available_date.isoformat(),
                }
                for window in evidence.source_windows
            ],
            hide_index=True,
            width="stretch",
        )

    if evidence.street_rankings:
        st.caption("Highest-value qualifying streets")
        st.dataframe(
            [
                {
                    "Rank": street.rank,
                    "Street": street.street,
                    "Median price": format_pounds(street.median_price_gbp),
                    "Transactions": street.transaction_count,
                }
                for street in evidence.street_rankings
            ],
            hide_index=True,
            width="stretch",
        )

    for chart in evidence.charts:
        render_chart(chart)

    if evidence.limitations:
        st.subheader("Limitations")
        for limitation in evidence.limitations:
            st.warning(limitation)


def render_chart(chart: ChartData) -> None:
    """Render canonical chart-ready data without persisting an image."""

    if not chart.labels or not chart.series:
        return

    frame = pd.DataFrame(
        {
            "Period": chart.labels,
            **{series.name: series.values for series in chart.series},
        }
    )
    series_names = [series.name for series in chart.series]

    st.caption(chart.title)
    if chart.chart_type == "bar":
        st.bar_chart(
            frame,
            x="Period",
            y=series_names,
            width="stretch",
        )
    else:
        st.line_chart(
            frame,
            x="Period",
            y=series_names,
            width="stretch",
        )


def render_note(state: WorkflowState) -> None:
    """Display the proposed grounded note and verification result."""

    draft = state.get("draft")
    if draft is None:
        return

    st.subheader("Proposed research note")
    st.write(draft)

    verification = state.get("verification")
    if verification is not None:
        if verification.supported:
            st.success(
                f"Verified {verification.checked_claim_count} numerical "
                "or date claims against the frozen evidence."
            )
        else:
            st.error("The note contains unsupported claims and was not accepted.")


def render_approval(
    state: WorkflowState,
    orchestrator: Orchestrator,
    current_user: UserContext,
) -> None:
    """Render approval controls for the exact frozen report."""

    if state.get("status") != "awaiting_approval":
        return

    payload = state["report_payload"]
    same_owner = current_user.user_id == payload.owner.user_id

    st.subheader("Approval required")
    st.caption("The report is frozen. Approval applies to this exact hash.")

    details = st.columns(3)
    details[0].text_input(
        "Report name",
        value=payload.report_name,
        disabled=True,
        key=f"approved-name-{state['run_id']}",
    )
    details[1].text_input(
        "Owner",
        value=payload.owner.display_name,
        disabled=True,
        key=f"approved-owner-{state['run_id']}",
    )
    details[2].text_input(
        "Destination",
        value=payload.destination,
        disabled=True,
        key=f"approved-destination-{state['run_id']}",
    )

    st.code(state["report_hash"], language=None)

    if not same_owner:
        st.error(
            "The selected demonstration identity does not own this run. "
            "Switch back to its owner to approve or reject it."
        )

    approve_column, reject_column = st.columns(2)
    if approve_column.button(
        "Approve and save exact report",
        type="primary",
        disabled=not same_owner,
    ):
        resume_with_decision(state, orchestrator, ApprovalChoice.APPROVE)

    if reject_column.button(
        "Reject report",
        disabled=not same_owner,
    ):
        resume_with_decision(state, orchestrator, ApprovalChoice.REJECT)


def resume_with_decision(
    state: WorkflowState,
    orchestrator: Orchestrator,
    choice: ApprovalChoice,
) -> None:
    """Resume the durable approval interrupt with a hash-bound decision."""

    decision = ApprovalDecision(
        run_id=state["run_id"],
        owner_id=state["user"].user_id,
        report_hash=state["report_hash"],
        choice=choice,
    )

    try:
        resumed = orchestrator.resume_approval(
            run_id=state["run_id"],
            decision=decision,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not resume the workflow: {exc}")
        return

    st.session_state["workflow_state"] = resumed
    st.rerun()


def render_audit_trace(state: WorkflowState) -> None:
    """Display ordered, human-readable audit events."""

    trace = state.get("audit_trace", ())
    if not trace:
        return

    with st.expander("Audit trace"):
        st.dataframe(
            [
                {
                    "Sequence": event.sequence,
                    "Time": event.timestamp.isoformat(),
                    "Action": event.action,
                    "Status": event.status.value,
                    "Explanation": event.explanation,
                }
                for event in trace
            ],
            hide_index=True,
            width="stretch",
        )


def render_saved_reports(
    repository: SQLiteReportRepository,
    user: UserContext,
) -> None:
    """List and open reports through owner-scoped repository queries."""

    st.divider()
    st.subheader("Saved reports for current owner")

    try:
        reports = repository.list_for_owner(user.user_id)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not list saved reports: {exc}")
        return

    if not reports:
        st.caption("No approved reports have been saved for this owner.")
        return

    report_ids = [report.report_id for report in reports]
    labels = {
        report.report_id: (
            f"{report.payload.report_name} — "
            f"{report.approved_at:%Y-%m-%d %H:%M}"
        )
        for report in reports
    }
    selected_id = st.selectbox(
        "Open report",
        options=report_ids,
        format_func=lambda report_id: labels[report_id],
        key=f"saved-report-{user.user_id}",
    )

    try:
        selected = repository.get_for_owner(selected_id, user.user_id)
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not open the selected report: {exc}")
        return

    if selected is None:
        st.error("The report does not exist for the current owner.")
        return

    render_saved_report(selected)


def render_saved_report(report: SavedReport) -> None:
    """Display and optionally download one owner-scoped report."""

    with st.expander(report.payload.report_name, expanded=True):
        st.write(report.payload.research_note)

        for chart in report.payload.charts:
            render_chart(chart)

        st.caption(
            f"Report ID: {report.report_id} · "
            f"Approved: {report.approved_at.isoformat()}"
        )
        st.download_button(
            "Download Markdown",
            data=report_markdown(report),
            file_name=f"{report.report_id}.md",
            mime="text/markdown",
            key=f"download-{report.report_id}",
        )


def report_markdown(report: SavedReport) -> str:
    """Build a portable Markdown view without changing canonical storage."""

    payload = report.payload
    limitations = "\n".join(
        f"- {limitation}" for limitation in payload.evidence.limitations
    ) or "- None recorded."

    return (
        f"# {payload.report_name}\n\n"
        f"**Report ID:** {report.report_id}  \n"
        f"**Owner:** {payload.owner.display_name}  \n"
        f"**Postcode:** {payload.postcode}  \n"
        f"**Approved:** {report.approved_at.isoformat()}  \n\n"
        f"## Research note\n\n{payload.research_note}\n\n"
        f"## Limitations\n\n{limitations}\n"
    )


def format_pounds(value: float | None) -> str:
    """Format an optional monetary value."""

    return "Unavailable" if value is None else f"£{value:,.0f}"


def format_percentage(value: float | None) -> str:
    """Format an optional percentage."""

    return "Suppressed" if value is None else f"{value:,.1f}%"


def main() -> None:
    """Run the Streamlit demonstration."""

    st.set_page_config(
        page_title="Land Registry Agent",
        page_icon="🏠",
        layout="wide",
    )
    st.title("Land Registry Agent")
    st.caption(
        "Deterministic analysis, grounded drafting and explicit approval "
        "before persistence."
    )

    try:
        services = create_services()
    except (OSError, RuntimeError, ValidationError, ValueError) as exc:
        st.error(f"Application configuration failed: {exc}")
        st.stop()

    user = render_demo_user(services.settings)

    st.sidebar.divider()
    st.sidebar.caption(
        f"Intent model: {services.settings.openai_intent_model}"
    )
    st.sidebar.caption(
        f"Draft model: {services.settings.openai_draft_model}"
    )

    if services.orchestrator is None:
        st.warning(
            "OPENAI_API_KEY is not configured. Existing owner-scoped reports "
            "can still be viewed, but new research runs are disabled."
        )

    state = render_request_form(services.orchestrator, user)
    if state is not None:
        status = state.get("status", "unknown")
        st.caption(f"Run status: {status}")

        if state.get("error"):
            if status == "clarification_required":
                st.warning(state["error"])
            else:
                st.error(state["error"])

        render_intent_summary(state)
        render_plan(state)

        if status == "planned" and services.orchestrator is not None:
            st.info(
                "The plan exists in the checkpoint. No external data source "
                "has been called yet."
            )
            if st.button("Execute plan", type="primary"):
                try:
                    with st.spinner(
                        "Retrieving and analysing the latest available data..."
                    ):
                        state = services.orchestrator.execute(state["run_id"])
                    st.session_state["workflow_state"] = state
                    st.rerun()
                except (OSError, RuntimeError, ValueError) as exc:
                    st.error(f"Could not execute the workflow: {exc}")

        evidence = state.get("evidence")
        if evidence is not None:
            render_evidence(evidence)

        render_note(state)

        if services.orchestrator is not None:
            render_approval(state, services.orchestrator, user)

        saved = state.get("saved_report")
        if saved is not None:
            st.success(
                f"Saved approved report {saved.report_id} for "
                f"{saved.payload.owner.display_name}."
            )
        elif status == "rejected":
            st.info("The report was rejected and nothing was written.")
        elif status == "completed":
            st.success("The requested analysis completed without persistence.")

        render_audit_trace(state)

    render_saved_reports(services.repository, user)


if __name__ == "__main__":
    main()
