"""Deterministic execution skills used to compose workflow plans."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from land_registry_agent.models import (
    ExecutionPlan,
    Intent,
    PersistenceMode,
    PlanStep,
    ToolCategory,
)

SkillPredicate = Callable[[Intent], bool]


@dataclass(frozen=True)
class SkillOperation:
    """One ordered operation contributed by an execution skill."""

    order: int
    operation: str
    category: ToolCategory
    description_template: str

    def build_step(self, intent: Intent, sequence: int) -> PlanStep:
        """Render this operation as a request-specific plan step."""

        description = self.description_template.format(
            postcode=intent.postcode or "the requested postcode",
            requested_years=intent.requested_years,
            region=intent.region or "the requested region",
        )
        return PlanStep(
            sequence=sequence,
            operation=self.operation,
            category=self.category,
            description=description,
        )


@dataclass(frozen=True)
class ExecutionSkill:
    """A bounded plan fragment activated from validated intent."""

    name: str
    description: str
    applies_when: SkillPredicate
    operations: tuple[SkillOperation, ...]

    def applies(self, intent: Intent) -> bool:
        """Return whether this skill applies to the supplied intent."""

        return self.applies_when(intent)


def _always(_: Intent) -> bool:
    return True


def _regional_comparison_requested(intent: Intent) -> bool:
    return intent.regional_comparison_requested


def _street_ranking_requested(intent: Intent) -> bool:
    return intent.street_ranking_requested


def _note_requested(intent: Intent) -> bool:
    return intent.note_requested


def _persistence_requested(intent: Intent) -> bool:
    return intent.persistence_mode is PersistenceMode.REQUESTED


LOCAL_PROPERTY_TRENDS = ExecutionSkill(
    name="local_property_trends",
    description="Retrieve and analyse local Price Paid evidence.",
    applies_when=_always,
    operations=(
        SkillOperation(
            order=10,
            operation="fetch_price_paid_transactions",
            category=ToolCategory.READ,
            description_template=(
                "Retrieve flat Price Paid transaction rows for {postcode} "
                "using the latest available {requested_years}-year window."
            ),
        ),
        SkillOperation(
            order=30,
            operation="analyse_local_trends",
            category=ToolCategory.ANALYSIS,
            description_template=(
                "Calculate monthly transaction counts, medians, means and "
                "local change."
            ),
        ),
        SkillOperation(
            order=60,
            operation="build_chart_data",
            category=ToolCategory.ANALYSIS,
            description_template=(
                "Prepare chart-ready time-series and qualifying street data."
            ),
        ),
    ),
)

REGIONAL_COMPARISON = ExecutionSkill(
    name="regional_comparison",
    description="Retrieve HPI evidence and compare overlapping trends.",
    applies_when=_regional_comparison_requested,
    operations=(
        SkillOperation(
            order=20,
            operation="fetch_regional_hpi",
            category=ToolCategory.READ,
            description_template=(
                "Retrieve the most recent available House Price Index "
                "records for {region}."
            ),
        ),
        SkillOperation(
            order=40,
            operation="compare_regional_hpi",
            category=ToolCategory.ANALYSIS,
            description_template=(
                "Compare local and regional trends only when their source "
                "periods support a like-for-like comparison."
            ),
        ),
    ),
)

STREET_VALUE_RANKING = ExecutionSkill(
    name="street_value_ranking",
    description="Rank qualifying streets from local evidence.",
    applies_when=_street_ranking_requested,
    operations=(
        SkillOperation(
            order=50,
            operation="rank_high_value_streets",
            category=ToolCategory.ANALYSIS,
            description_template=(
                "Rank qualifying streets using a minimum of three transactions."
            ),
        ),
    ),
)

RESEARCH_NOTE = ExecutionSkill(
    name="research_note",
    description="Draft and verify the requested grounded research note.",
    applies_when=_note_requested,
    operations=(
        SkillOperation(
            order=70,
            operation="draft_research_note",
            category=ToolCategory.LANGUAGE,
            description_template=(
                "Draft the requested bounded research note from the frozen "
                "evidence bundle."
            ),
        ),
        SkillOperation(
            order=80,
            operation="verify_research_note",
            category=ToolCategory.ANALYSIS,
            description_template=(
                "Verify numerical and date claims against the evidence "
                "allowlist."
            ),
        ),
    ),
)

APPROVED_PERSISTENCE = ExecutionSkill(
    name="approved_persistence",
    description="Pause for approval and persist only the approved payload.",
    applies_when=_persistence_requested,
    operations=(
        SkillOperation(
            order=90,
            operation="request_approval",
            category=ToolCategory.WRITE,
            description_template=(
                "Pause and request approval for the exact frozen report payload."
            ),
        ),
        SkillOperation(
            order=100,
            operation="save_approved_report",
            category=ToolCategory.WRITE,
            description_template=(
                "Persist only the exact approved and hash-validated payload."
            ),
        ),
    ),
)

DEFAULT_EXECUTION_SKILLS = (
    LOCAL_PROPERTY_TRENDS,
    REGIONAL_COMPARISON,
    STREET_VALUE_RANKING,
    RESEARCH_NOTE,
    APPROVED_PERSISTENCE,
)


def select_execution_skills(
    intent: Intent,
    skills: Sequence[ExecutionSkill] = DEFAULT_EXECUTION_SKILLS,
) -> tuple[ExecutionSkill, ...]:
    """Select applicable skills deterministically from validated intent."""

    return tuple(skill for skill in skills if skill.applies(intent))


def compose_execution_plan(
    run_id: str,
    intent: Intent,
    skills: Sequence[ExecutionSkill] = DEFAULT_EXECUTION_SKILLS,
) -> ExecutionPlan:
    """Compose selected skill operations into one ordered typed plan."""

    selected = select_execution_skills(intent, skills)
    if not selected:
        raise ValueError("At least one execution skill must apply.")

    skill_names = [skill.name for skill in selected]
    if len(set(skill_names)) != len(skill_names):
        raise ValueError("Execution skill names must be unique.")

    operations = [
        operation
        for skill in selected
        for operation in skill.operations
    ]

    operation_names = [operation.operation for operation in operations]
    if len(set(operation_names)) != len(operation_names):
        raise ValueError("Execution skills contributed duplicate operations.")

    orders = [operation.order for operation in operations]
    if any(order <= 0 for order in orders):
        raise ValueError("Skill operation ordering values must be positive.")
    if len(set(orders)) != len(orders):
        raise ValueError("Execution skills contributed conflicting order values.")

    ordered = sorted(operations, key=lambda operation: operation.order)
    steps = tuple(
        operation.build_step(intent, sequence)
        for sequence, operation in enumerate(ordered, start=1)
    )

    return ExecutionPlan(
        run_id=run_id,
        selected_skills=tuple(skill_names),
        steps=steps,
    )
