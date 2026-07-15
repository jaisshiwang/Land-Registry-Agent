"""Opt-in smoke tests for live data sources and configured OpenAI models."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import date

import pytest
from dotenv import load_dotenv

from land_registry_agent.config import Settings
from land_registry_agent.data import LandRegistryGateway
from land_registry_agent.llm import diagnose_model_access

load_dotenv(override=False)

LIVE_TESTS_ENABLED = (
    os.getenv("RUN_LIVE_TESTS", "").strip().lower()
    in {"1", "true", "yes"}
)


@pytest.fixture(scope="module")
def live_settings() -> Settings:
    """Load bounded live-test settings or skip unless explicitly enabled."""

    if not LIVE_TESTS_ENABLED:
        pytest.skip(
            "Set RUN_LIVE_TESTS=1 to run live connection tests."
        )

    return Settings(
        cache_ttl_seconds=0,
        query_page_size=100,
        query_max_pages=5,
        http_timeout_seconds=60,
        http_max_retries=2,
    )


@pytest.fixture(scope="module")
def live_gateway(
    live_settings: Settings,
) -> Iterator[LandRegistryGateway]:
    """Yield a real gateway and close its owned HTTP client afterwards."""

    gateway = LandRegistryGateway(live_settings)
    try:
        yield gateway
    finally:
        gateway.close()


def test_price_paid_api_connection(
    live_gateway: LandRegistryGateway,
) -> None:
    """Fetch a bounded full-postcode Price Paid sample."""

    result = live_gateway.fetch_price_paid_transactions(
        postcode="GU1 1AA",
        requested_years=1,
    )

    assert result.source_url.endswith("/landregistry/query")
    assert result.artifact_keys

    for transaction in result.records:
        assert transaction.postcode == "GU1 1AA"
        assert transaction.price_gbp > 0
        assert transaction.property_type in {
            "detached",
            "semi_detached",
            "terraced",
            "flat_maisonette",
            "other",
            "unknown",
        }


def test_regional_hpi_api_connection(
    live_gateway: LandRegistryGateway,
) -> None:
    """Fetch the latest available South East HPI window."""

    result = live_gateway.fetch_regional_hpi(
        region="South East",
        requested_years=1,
    )

    assert result.source_url.endswith(
        "/data/hpi/region/south-east.json"
    )
    assert result.artifact_keys
    assert result.records
    assert result.records == tuple(
        sorted(result.records, key=lambda record: record.period)
    )
    assert result.records[-1].period <= date.today()
    assert all(
        record.region == "South East"
        for record in result.records
    )


def test_configured_openai_model_access(
    live_settings: Settings,
) -> None:
    """Check configured model metadata using the supplied API key."""

    if not live_settings.openai_enabled:
        pytest.fail(
            "OPENAI_API_KEY is required when RUN_LIVE_TESTS=1."
        )

    results = diagnose_model_access(live_settings)

    assert {result.service_name for result in results} == {
        "intent",
        "draft",
    }

    failures = [
        (
            f"{result.service_name} model "
            f"{result.model_name!r}: {result.explanation}"
        )
        for result in results
        if not result.accessible
    ]
    assert failures == [], "\n".join(failures)
