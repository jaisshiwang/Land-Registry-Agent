"""Tests for request-specific planning and capability gating."""

from __future__ import annotations

import pytest

from land_registry_agent.models import (
    Intent,
    InterpretationMethod,
    PersistenceMode,
    StepStatus,
    ToolCategory,
)
from land_registry_agent.orchestrator import (
    ToolCapability,
    ToolRegistry,
    build_execution_plan,
)
from land_registry_agent.skills import (
    ExecutionSkill,
    SkillOperation,
    compose_execution_plan,
)


def make_intent(**updates: object) -> Intent:
    """Build a valid intent with concise test-specific overrides."""

    base = Intent(
        postcode="GU1",
        requested_years=3,
        regional_comparison_requested=False,
        street_ranking_requested=False,
        note_requested=True,
        persistence_mode=PersistenceMode.NOT_REQUESTED,
        latest_available_data=True,
        live_refresh_requested=False,
        interpretation_method=InterpretationMethod.DETERMINISTIC,
        confidence=0.95,
    )
    return base.model_copy(update=updates)


def operations(intent: Intent) -> list[str]:
    """Return ordered operation names from a generated plan."""

    plan = build_execution_plan("run_test", intent)
    return [step.operation for step in plan.steps]


def test_target_request_generates_complete_ordered_plan() -> None:
    intent = make_intent(
        region="South East",
        regional_comparison_requested=True,
        street_ranking_requested=True,
        persistence_mode=PersistenceMode.REQUESTED,
    )

    plan = build_execution_plan("run_target", intent)

    assert plan.run_id == "run_target"
    assert [step.sequence for step in plan.steps] == list(
        range(1, len(plan.steps) + 1)
    )
    assert all(step.status is StepStatus.PENDING for step in plan.steps)
    assert plan.selected_skills == (
        "local_property_trends",
        "regional_comparison",
        "street_value_ranking",
        "research_note",
        "approved_persistence",
    )
    assert [step.operation for step in plan.steps] == [
        "fetch_price_paid_transactions",
        "fetch_regional_hpi",
        "analyse_local_trends",
        "compare_regional_hpi",
        "rank_high_value_streets",
        "build_chart_data",
        "draft_research_note",
        "verify_research_note",
        "request_approval",
        "save_approved_report",
    ]


def test_analysis_only_plan_omits_optional_operations() -> None:
    intent = make_intent(
        note_requested=False,
        persistence_mode=PersistenceMode.NOT_REQUESTED,
    )

    plan = build_execution_plan("run_analysis_only", intent)

    assert plan.selected_skills == ("local_property_trends",)
    assert [step.operation for step in plan.steps] == [
        "fetch_price_paid_transactions",
        "analyse_local_trends",
        "build_chart_data",
    ]


def test_regional_comparison_adds_only_regional_operations() -> None:
    intent = make_intent(
        region="London",
        regional_comparison_requested=True,
    )

    planned_operations = operations(intent)

    assert "fetch_regional_hpi" in planned_operations
    assert "compare_regional_hpi" in planned_operations
    assert "rank_high_value_streets" not in planned_operations
    assert "request_approval" not in planned_operations
    assert "save_approved_report" not in planned_operations


def test_street_ranking_is_independent_of_regional_comparison() -> None:
    intent = make_intent(street_ranking_requested=True)

    planned_operations = operations(intent)

    assert "rank_high_value_streets" in planned_operations
    assert "fetch_regional_hpi" not in planned_operations
    assert "compare_regional_hpi" not in planned_operations


def test_forbidden_persistence_omits_approval_and_write() -> None:
    intent = make_intent(
        persistence_mode=PersistenceMode.FORBIDDEN,
    )

    planned_operations = operations(intent)

    assert "draft_research_note" in planned_operations
    assert "verify_research_note" in planned_operations
    assert "request_approval" not in planned_operations
    assert "save_approved_report" not in planned_operations


def test_plan_categories_separate_reads_analysis_language_and_writes() -> None:
    intent = make_intent(
        region="South East",
        regional_comparison_requested=True,
        persistence_mode=PersistenceMode.REQUESTED,
    )

    plan = build_execution_plan("run_categories", intent)
    categories = {step.operation: step.category for step in plan.steps}

    assert categories["fetch_price_paid_transactions"] is ToolCategory.READ
    assert categories["analyse_local_trends"] is ToolCategory.ANALYSIS
    assert categories["draft_research_note"] is ToolCategory.LANGUAGE
    assert categories["save_approved_report"] is ToolCategory.WRITE


def test_registry_blocks_write_capability_without_approval() -> None:
    writes: list[str] = []

    def save_report() -> str:
        writes.append("saved")
        return "report-id"

    registry = ToolRegistry(
        (
            ToolCapability(
                name="save_approved_report",
                category=ToolCategory.WRITE,
                handler=save_report,
            ),
        )
    )

    with pytest.raises(
        PermissionError,
        match="requires explicit approval",
    ):
        registry.invoke("save_approved_report")

    assert writes == []

    result = registry.invoke(
        "save_approved_report",
        approval_granted=True,
    )

    assert result == "report-id"
    assert writes == ["saved"]


def test_registry_rejects_duplicate_capability_names() -> None:
    capability = ToolCapability(
        name="duplicate",
        category=ToolCategory.READ,
        handler=lambda: None,
    )

    with pytest.raises(
        ValueError,
        match="Capability names must be unique",
    ):
        ToolRegistry((capability, capability))


def test_skill_composer_rejects_duplicate_operation_names() -> None:
    first = ExecutionSkill(
        name="first",
        description="First test skill.",
        applies_when=lambda _: True,
        operations=(
            SkillOperation(
                order=10,
                operation="duplicate_operation",
                category=ToolCategory.READ,
                description_template="First operation.",
            ),
        ),
    )
    second = ExecutionSkill(
        name="second",
        description="Second test skill.",
        applies_when=lambda _: True,
        operations=(
            SkillOperation(
                order=20,
                operation="duplicate_operation",
                category=ToolCategory.ANALYSIS,
                description_template="Second operation.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="duplicate operations"):
        compose_execution_plan("run_duplicate", make_intent(), (first, second))


def test_skill_composer_rejects_conflicting_order_values() -> None:
    first = ExecutionSkill(
        name="first",
        description="First test skill.",
        applies_when=lambda _: True,
        operations=(
            SkillOperation(
                order=10,
                operation="first_operation",
                category=ToolCategory.READ,
                description_template="First operation.",
            ),
        ),
    )
    second = ExecutionSkill(
        name="second",
        description="Second test skill.",
        applies_when=lambda _: True,
        operations=(
            SkillOperation(
                order=10,
                operation="second_operation",
                category=ToolCategory.ANALYSIS,
                description_template="Second operation.",
            ),
        ),
    )

    with pytest.raises(ValueError, match="conflicting order"):
        compose_execution_plan("run_conflict", make_intent(), (first, second))


def test_skill_composer_requires_at_least_one_applicable_skill() -> None:
    with pytest.raises(ValueError, match="At least one"):
        compose_execution_plan("run_empty", make_intent(), ())
