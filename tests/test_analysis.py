"""Tests for deterministic property analysis and evidence policy."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from land_registry_agent.analysis import PropertyAnalyzer
from land_registry_agent.config import Settings
from land_registry_agent.models import (
    EvidenceBundle,
    HPIRecord,
    Intent,
    InterpretationMethod,
    PersistenceMode,
    Transaction,
)


def make_intent(**updates: object) -> Intent:
    """Build a valid analytical intent."""

    intent = Intent(
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
    return intent.model_copy(update=updates)


def make_transaction(
    number: int,
    *,
    month: int,
    price: int,
    street: str,
) -> Transaction:
    """Build one normalised transaction."""

    return Transaction(
        transaction_id=f"transaction-{number}",
        transfer_date=date(2024, month, min(number, 28)),
        price_gbp=price,
        postcode="GU1 1AA",
        property_type="terraced",
        street=street,
    )


def build_evidence(
    analyzer: PropertyAnalyzer,
    intent: Intent,
    transactions: Sequence[Transaction],
    hpi_records: Sequence[HPIRecord] = (),
) -> EvidenceBundle:
    """Build evidence with stable test provenance."""

    return analyzer.build_evidence(
        user_request="Test request",
        intent=intent,
        transactions=transactions,
        hpi_records=hpi_records,
        price_paid_source_url="https://example.test/price-paid",
        price_paid_artifact_keys=("price-artifact",),
        hpi_source_url=(
            "https://example.test/hpi"
            if hpi_records
            else None
        ),
        hpi_artifact_keys=(
            ("hpi-artifact",)
            if hpi_records
            else ()
        ),
    )


def test_monthly_counts_medians_means_and_chart_schema() -> None:
    analyzer = PropertyAnalyzer(
        Settings(minimum_local_transactions=1)
    )
    transactions = (
        make_transaction(1, month=1, price=100_000, street="Alpha Road"),
        make_transaction(2, month=1, price=300_000, street="Alpha Road"),
        make_transaction(3, month=2, price=200_000, street="Beta Road"),
        make_transaction(4, month=2, price=400_000, street="Beta Road"),
    )

    evidence = build_evidence(
        analyzer,
        make_intent(),
        transactions,
    )

    assert len(evidence.monthly_local_metrics) == 2

    january, february = evidence.monthly_local_metrics
    assert january.period == date(2024, 1, 1)
    assert january.transaction_count == 2
    assert january.median_price_gbp == 200_000
    assert january.mean_price_gbp == 200_000

    assert february.period == date(2024, 2, 1)
    assert february.transaction_count == 2
    assert february.median_price_gbp == 300_000
    assert february.mean_price_gbp == 300_000

    assert evidence.local_trend.start_value == 200_000
    assert evidence.local_trend.end_value == 300_000
    assert evidence.local_trend.percentage_change == 50.0
    assert evidence.local_trend.change_claim_permitted is True

    local_chart = next(
        chart
        for chart in evidence.charts
        if chart.chart_id == "local-monthly-prices"
    )
    assert local_chart.chart_type == "line"
    assert local_chart.labels == ("2024-01", "2024-02")
    assert len(local_chart.series) == 2
    assert all(
        len(series.values) == len(local_chart.labels)
        for series in local_chart.series
    )
    assert local_chart.series[0].sample_sizes == (2, 2)


def test_sparse_data_suppresses_change_and_street_ranking() -> None:
    analyzer = PropertyAnalyzer(
        Settings(
            minimum_local_transactions=10,
            minimum_street_transactions=3,
        )
    )
    transactions = tuple(
        make_transaction(
            number,
            month=1 if number <= 5 else 2,
            price=200_000 + number * 1_000,
            street="Sparse Street",
        )
        for number in range(1, 10)
    )

    evidence = build_evidence(
        analyzer,
        make_intent(street_ranking_requested=True),
        transactions,
    )

    assert evidence.confidence.value == "low"
    assert evidence.local_trend.change_claim_permitted is False
    assert evidence.local_trend.percentage_change is None
    assert evidence.street_rankings == ()
    assert any(
        "only 9 transactions" in limitation
        for limitation in evidence.limitations
    )
    assert all(
        chart.chart_id != "highest-value-streets"
        for chart in evidence.charts
    )


def test_street_ranking_requires_three_transactions() -> None:
    analyzer = PropertyAnalyzer(
        Settings(
            minimum_local_transactions=10,
            minimum_street_transactions=3,
        )
    )
    transactions = (
        make_transaction(1, month=1, price=500_000, street="Alpha Road"),
        make_transaction(2, month=1, price=510_000, street="Alpha Road"),
        make_transaction(3, month=1, price=520_000, street="Alpha Road"),
        make_transaction(4, month=1, price=900_000, street="Excluded Lane"),
        make_transaction(5, month=1, price=950_000, street="Excluded Lane"),
        make_transaction(6, month=1, price=300_000, street="Charlie Street"),
        make_transaction(7, month=2, price=310_000, street="Charlie Street"),
        make_transaction(8, month=2, price=320_000, street="Charlie Street"),
        make_transaction(9, month=2, price=330_000, street="Charlie Street"),
        make_transaction(10, month=2, price=340_000, street="Charlie Street"),
        make_transaction(11, month=2, price=350_000, street="Charlie Street"),
        make_transaction(12, month=2, price=360_000, street="Charlie Street"),
    )

    evidence = build_evidence(
        analyzer,
        make_intent(street_ranking_requested=True),
        transactions,
    )

    assert [item.street for item in evidence.street_rankings] == [
        "Alpha Road",
        "Charlie Street",
    ]
    assert evidence.street_rankings[0].median_price_gbp == 510_000
    assert evidence.street_rankings[0].transaction_count == 3
    assert all(
        item.street != "Excluded Lane"
        for item in evidence.street_rankings
    )

    street_chart = next(
        chart
        for chart in evidence.charts
        if chart.chart_id == "highest-value-streets"
    )
    assert street_chart.chart_type == "bar"
    assert street_chart.labels == ("Alpha Road", "Charlie Street")


def test_non_overlapping_source_periods_prohibit_comparison() -> None:
    analyzer = PropertyAnalyzer(
        Settings(minimum_local_transactions=10)
    )
    transactions = tuple(
        make_transaction(
            number,
            month=1 if number <= 5 else 2,
            price=250_000 + number * 2_000,
            street="Local Road",
        )
        for number in range(1, 11)
    )
    hpi_records = (
        HPIRecord(
            period=date(2016, 1, 1),
            region="South East",
            average_price_gbp=300_000,
        ),
        HPIRecord(
            period=date(2016, 2, 1),
            region="South East",
            average_price_gbp=303_000,
        ),
    )

    evidence = build_evidence(
        analyzer,
        make_intent(
            region="South East",
            regional_comparison_requested=True,
        ),
        transactions,
        hpi_records,
    )

    assert evidence.periods_overlap is False
    assert evidence.regional_comparison_claim_permitted is False
    assert evidence.comparison_difference_percentage_points is None
    assert len(evidence.source_windows) == 2
    assert any(
        "do not overlap" in limitation
        for limitation in evidence.limitations
    )
    assert any(
        "historically stale" in limitation
        for limitation in evidence.limitations
    )
    assert all(
        chart.chart_id != "local-regional-normalised"
        for chart in evidence.charts
    )
