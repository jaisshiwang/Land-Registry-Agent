"""Deterministic analysis, evidence construction, hashing, and verification."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import date
from typing import Any, cast

import pandas as pd
from pydantic import BaseModel

from land_registry_agent.config import Settings
from land_registry_agent.models import (
    ChartData,
    ChartSeries,
    ConfidenceLevel,
    EvidenceBundle,
    HPIRecord,
    Intent,
    MonthlyMetric,
    SourceWindow,
    StreetMetric,
    Transaction,
    TrendSummary,
    VerificationResult,
)

DATE_CLAIM_PATTERN = re.compile(
    r"\b\d{4}-\d{2}(?:-\d{2})?\b"
    r"|\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b",
    re.IGNORECASE,
)

NUMBER_CLAIM_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])£?\s*-?\d[\d,]*(?:\.\d+)?%?"
)

PERFORMANCE_PATTERN = re.compile(
    r"\b(?:outperform(?:ed|ing)?|underperform(?:ed|ing)?|"
    r"faster than|slower than|ahead of|behind)\b",
    re.IGNORECASE,
)

STREET_SUPERLATIVE_PATTERN = re.compile(
    r"\b(?:highest[- ]value street|most expensive street|top street)\b",
    re.IGNORECASE,
)


class PropertyAnalyzer:
    """Apply deterministic analytical and evidence policies."""

    def __init__(self, settings: Settings) -> None:
        self._minimum_local_transactions = settings.minimum_local_transactions
        self._minimum_street_transactions = settings.minimum_street_transactions

    def build_evidence(
        self,
        *,
        user_request: str,
        intent: Intent,
        transactions: Sequence[Transaction],
        hpi_records: Sequence[HPIRecord] = (),
        price_paid_source_url: str,
        price_paid_artifact_keys: Sequence[str] = (),
        hpi_source_url: str | None = None,
        hpi_artifact_keys: Sequence[str] = (),
    ) -> EvidenceBundle:
        """Calculate all evidence from normalized source records."""

        transaction_frame = _transaction_frame(transactions)
        monthly_metrics = _monthly_metrics(transaction_frame)

        transaction_count = len(transaction_frame)
        local_change_permitted = (
            transaction_count >= self._minimum_local_transactions
            and len(monthly_metrics) >= 2
        )
        local_trend = _local_trend(
            intent,
            monthly_metrics,
            local_change_permitted,
        )

        limitations: list[str] = []
        source_windows: list[SourceWindow] = []

        local_window = _transaction_window(transactions)
        if local_window is not None:
            source_windows.append(local_window)

        if transaction_count == 0:
            limitations.append(
                "No Price Paid transactions were returned for the requested window."
            )
        elif transaction_count < self._minimum_local_transactions:
            limitations.append(
                "Local percentage-change and street-ranking claims were suppressed "
                f"because only {transaction_count} transactions were available; "
                f"at least {self._minimum_local_transactions} are required."
            )
        elif len(monthly_metrics) < 2:
            limitations.append(
                "Local percentage change was suppressed because fewer than two "
                "monthly observations were available."
            )

        street_rankings = self._rank_streets(
            intent=intent,
            frame=transaction_frame,
            local_evidence_sufficient=local_change_permitted,
        )
        if (
            intent.street_ranking_requested
            and local_change_permitted
            and not street_rankings
        ):
            limitations.append(
                "Street ranking was suppressed because no street met the minimum "
                f"sample of {self._minimum_street_transactions} transactions."
            )

        sorted_hpi = tuple(sorted(hpi_records, key=lambda item: item.period))
        regional_trend = _regional_trend(intent, sorted_hpi)
        hpi_window = _hpi_window(sorted_hpi)

        if hpi_window is not None:
            source_windows.append(hpi_window)
            if (date.today() - hpi_window.latest_available_date).days > 365:
                limitations.append(
                    "Regional HPI data is historically stale: the latest available "
                    f"period is {hpi_window.latest_available_date:%Y-%m}."
                )
        elif intent.regional_comparison_requested:
            limitations.append(
                "Regional comparison was unavailable because no HPI records "
                "were returned."
            )

        (
            periods_overlap,
            comparison_permitted,
            comparison_difference,
        ) = _regional_comparison(
            intent=intent,
            monthly_metrics=monthly_metrics,
            hpi_records=sorted_hpi,
            local_window=local_window,
            hpi_window=hpi_window,
            minimum_local_transactions=self._minimum_local_transactions,
        )

        if intent.regional_comparison_requested and periods_overlap is False:
            limitations.append(
                "Local and regional source periods do not overlap, so a "
                "like-for-like performance claim is prohibited."
            )
        elif (
            intent.regional_comparison_requested
            and periods_overlap
            and not comparison_permitted
        ):
            limitations.append(
                "Regional comparison was suppressed because the overlapping "
                "period did not contain sufficient evidence."
            )

        confidence = _confidence_level(
            transaction_count,
            self._minimum_local_transactions,
        )

        charts = _build_charts(
            monthly_metrics=monthly_metrics,
            street_rankings=street_rankings,
            hpi_records=sorted_hpi,
            comparison_permitted=comparison_permitted,
        )

        source_urls = [price_paid_source_url]
        if hpi_source_url is not None:
            source_urls.append(hpi_source_url)

        return EvidenceBundle(
            user_request=user_request,
            intent=intent,
            source_windows=tuple(source_windows),
            monthly_local_metrics=monthly_metrics,
            local_trend=local_trend,
            regional_trend=regional_trend,
            periods_overlap=periods_overlap,
            regional_comparison_claim_permitted=comparison_permitted,
            comparison_difference_percentage_points=comparison_difference,
            street_rankings=street_rankings,
            confidence=confidence,
            limitations=tuple(dict.fromkeys(limitations)),
            charts=charts,
            source_urls=tuple(dict.fromkeys(source_urls)),
            artifact_keys=tuple(
                dict.fromkeys(
                    (*price_paid_artifact_keys, *hpi_artifact_keys)
                )
            ),
        )

    def _rank_streets(
        self,
        *,
        intent: Intent,
        frame: pd.DataFrame,
        local_evidence_sufficient: bool,
    ) -> tuple[StreetMetric, ...]:
        """Rank streets only when requested and supported."""

        if (
            not intent.street_ranking_requested
            or not local_evidence_sufficient
            or frame.empty
        ):
            return ()

        street_frame = frame.dropna(subset=["street"]).copy()
        street_frame["street"] = street_frame["street"].astype(str).str.strip()
        street_frame = street_frame[street_frame["street"] != ""]

        if street_frame.empty:
            return ()

        grouped = (
            street_frame.groupby("street", as_index=False)["price_gbp"]
            .agg(["count", "median"])
            .reset_index()
        )
        qualifying = grouped[
            grouped["count"] >= self._minimum_street_transactions
        ].copy()
        qualifying = qualifying.sort_values(
            by=["median", "street"],
            ascending=[False, True],
        ).head(5)

        records = qualifying.to_dict(orient="records")
        return tuple(
            StreetMetric(
                rank=index,
                street=str(record["street"]),
                median_price_gbp=float(record["median"]),
                transaction_count=int(record["count"]),
            )
            for index, record in enumerate(records, start=1)
        )


def canonical_json(model: BaseModel) -> str:
    """Serialize a model deterministically for hashing and persistence."""

    return json.dumps(
        model.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def content_hash(model: BaseModel) -> str:
    """Return a SHA-256 hash of a model's canonical representation."""

    return hashlib.sha256(canonical_json(model).encode("utf-8")).hexdigest()


def verify_note(note: str, evidence: EvidenceBundle) -> VerificationResult:
    """Fail closed when a note contains unsupported dates or numbers."""

    unsupported: list[str] = []
    checked_claims = 0

    allowed_dates = _allowed_date_claims(evidence)
    date_claims = DATE_CLAIM_PATTERN.findall(note)
    checked_claims += len(date_claims)

    for claim in date_claims:
        if claim.casefold() not in allowed_dates:
            unsupported.append(f"Unsupported date: {claim}")

    note_without_dates = DATE_CLAIM_PATTERN.sub(" ", note)
    number_claims = NUMBER_CLAIM_PATTERN.findall(note_without_dates)
    allowed_numbers = _collect_numbers(
        evidence.model_dump(mode="json")
    )
    checked_claims += len(number_claims)

    for claim in number_claims:
        if not _number_is_allowed(claim, allowed_numbers):
            unsupported.append(f"Unsupported number: {claim.strip()}")

    if PERFORMANCE_PATTERN.search(note):
        checked_claims += 1
        if not evidence.regional_comparison_claim_permitted:
            unsupported.append(
                "Unsupported local-versus-regional performance claim."
            )

    if STREET_SUPERLATIVE_PATTERN.search(note):
        checked_claims += 1
        if not evidence.street_rankings:
            unsupported.append("Unsupported street-ranking claim.")

    return VerificationResult(
        supported=not unsupported,
        checked_claim_count=checked_claims,
        unsupported_claims=tuple(dict.fromkeys(unsupported)),
    )


def _transaction_frame(
    transactions: Sequence[Transaction],
) -> pd.DataFrame:
    """Create the compact pandas frame used by local calculations."""

    return pd.DataFrame(
        [
            {
                "transfer_date": pd.Timestamp(item.transfer_date),
                "price_gbp": item.price_gbp,
                "street": item.street,
            }
            for item in transactions
        ],
        columns=["transfer_date", "price_gbp", "street"],
    )


def _monthly_metrics(frame: pd.DataFrame) -> tuple[MonthlyMetric, ...]:
    """Calculate monthly count, median, and mean sale price."""

    if frame.empty:
        return ()

    working = frame.copy()
    working["period"] = (
        working["transfer_date"].dt.to_period("M").dt.to_timestamp()
    )
    grouped = (
        working.groupby("period")["price_gbp"]
        .agg(["count", "median", "mean"])
        .reset_index()
        .sort_values("period")
    )

    records = grouped.to_dict(orient="records")
    return tuple(
        MonthlyMetric(
            period=cast(pd.Timestamp, record["period"]).date(),
            transaction_count=int(record["count"]),
            median_price_gbp=round(float(record["median"]), 2),
            mean_price_gbp=round(float(record["mean"]), 2),
        )
        for record in records
    )


def _local_trend(
    intent: Intent,
    monthly_metrics: Sequence[MonthlyMetric],
    claim_permitted: bool,
) -> TrendSummary:
    """Build the local median-price trend summary."""

    if not monthly_metrics:
        return TrendSummary(
            label=f"{intent.postcode or 'Local'} monthly median sale price",
            change_claim_permitted=False,
        )

    start_value = monthly_metrics[0].median_price_gbp
    end_value = monthly_metrics[-1].median_price_gbp
    percentage_change = (
        _percentage_change(start_value, end_value)
        if claim_permitted
        else None
    )

    return TrendSummary(
        label=f"{intent.postcode or 'Local'} monthly median sale price",
        start_value=start_value,
        end_value=end_value,
        percentage_change=percentage_change,
        change_claim_permitted=claim_permitted,
    )


def _regional_trend(
    intent: Intent,
    records: Sequence[HPIRecord],
) -> TrendSummary | None:
    """Build the latest available regional average-price trend."""

    if not records:
        return None

    start_value = records[0].average_price_gbp
    end_value = records[-1].average_price_gbp
    claim_permitted = len(records) >= 2

    return TrendSummary(
        label=f"{intent.region or records[0].region} regional average price",
        start_value=start_value,
        end_value=end_value,
        percentage_change=(
            _percentage_change(start_value, end_value)
            if claim_permitted
            else None
        ),
        change_claim_permitted=claim_permitted,
    )


def _transaction_window(
    transactions: Sequence[Transaction],
) -> SourceWindow | None:
    """Derive the actual Price Paid source window."""

    if not transactions:
        return None

    dates = [item.transfer_date for item in transactions]
    return SourceWindow(
        source_name="HM Land Registry Price Paid Data",
        start_date=min(dates),
        end_date=max(dates),
        latest_available_date=max(dates),
    )


def _hpi_window(
    records: Sequence[HPIRecord],
) -> SourceWindow | None:
    """Derive the actual HPI source window."""

    if not records:
        return None

    dates = [item.period for item in records]
    return SourceWindow(
        source_name="UK House Price Index",
        start_date=min(dates),
        end_date=max(dates),
        latest_available_date=max(dates),
    )


def _regional_comparison(
    *,
    intent: Intent,
    monthly_metrics: Sequence[MonthlyMetric],
    hpi_records: Sequence[HPIRecord],
    local_window: SourceWindow | None,
    hpi_window: SourceWindow | None,
    minimum_local_transactions: int,
) -> tuple[bool | None, bool, float | None]:
    """Calculate a like-for-like comparison only over an overlap."""

    if not intent.regional_comparison_requested:
        return None, False, None

    if local_window is None or hpi_window is None:
        return False, False, None

    overlap_start = max(local_window.start_date, hpi_window.start_date)
    overlap_end = min(local_window.end_date, hpi_window.end_date)
    periods_overlap = overlap_start <= overlap_end

    if not periods_overlap:
        return False, False, None

    local_overlap = [
        point
        for point in monthly_metrics
        if overlap_start <= point.period <= overlap_end
    ]
    hpi_overlap = [
        point
        for point in hpi_records
        if overlap_start <= point.period <= overlap_end
    ]
    overlapping_transactions = sum(
        point.transaction_count for point in local_overlap
    )

    comparison_permitted = (
        len(local_overlap) >= 2
        and len(hpi_overlap) >= 2
        and overlapping_transactions >= minimum_local_transactions
    )
    if not comparison_permitted:
        return True, False, None

    local_change = _percentage_change(
        local_overlap[0].median_price_gbp,
        local_overlap[-1].median_price_gbp,
    )
    regional_change = _percentage_change(
        hpi_overlap[0].average_price_gbp,
        hpi_overlap[-1].average_price_gbp,
    )

    return True, True, round(local_change - regional_change, 2)


def _build_charts(
    *,
    monthly_metrics: Sequence[MonthlyMetric],
    street_rankings: Sequence[StreetMetric],
    hpi_records: Sequence[HPIRecord],
    comparison_permitted: bool,
) -> tuple[ChartData, ...]:
    """Create renderer-independent chart data."""

    charts: list[ChartData] = []

    if monthly_metrics:
        charts.append(
            ChartData(
                chart_id="local-monthly-prices",
                chart_type="line",
                title="Monthly local sale prices",
                labels=tuple(
                    point.period.strftime("%Y-%m")
                    for point in monthly_metrics
                ),
                series=(
                    ChartSeries(
                        name="Median sale price",
                        values=tuple(
                            point.median_price_gbp
                            for point in monthly_metrics
                        ),
                        sample_sizes=tuple(
                            point.transaction_count
                            for point in monthly_metrics
                        ),
                    ),
                    ChartSeries(
                        name="Mean sale price",
                        values=tuple(
                            point.mean_price_gbp
                            for point in monthly_metrics
                        ),
                        sample_sizes=tuple(
                            point.transaction_count
                            for point in monthly_metrics
                        ),
                    ),
                ),
            )
        )

    if street_rankings:
        charts.append(
            ChartData(
                chart_id="highest-value-streets",
                chart_type="bar",
                title="Highest-value qualifying streets",
                labels=tuple(item.street for item in street_rankings),
                series=(
                    ChartSeries(
                        name="Median sale price",
                        values=tuple(
                            item.median_price_gbp
                            for item in street_rankings
                        ),
                        sample_sizes=tuple(
                            item.transaction_count
                            for item in street_rankings
                        ),
                    ),
                ),
            )
        )

    if comparison_permitted:
        local_by_period = {
            point.period: point.median_price_gbp
            for point in monthly_metrics
        }
        regional_by_period = {
            point.period: point.average_price_gbp
            for point in hpi_records
        }
        common_periods = sorted(
            set(local_by_period).intersection(regional_by_period)
        )

        if len(common_periods) >= 2:
            local_values = [local_by_period[period] for period in common_periods]
            regional_values = [
                regional_by_period[period] for period in common_periods
            ]
            charts.append(
                ChartData(
                    chart_id="local-regional-normalised",
                    chart_type="line",
                    title="Local versus regional price movement",
                    labels=tuple(
                        period.strftime("%Y-%m")
                        for period in common_periods
                    ),
                    series=(
                        ChartSeries(
                            name="Local median, first month = 100",
                            values=_normalise_series(local_values),
                        ),
                        ChartSeries(
                            name="Regional average, first month = 100",
                            values=_normalise_series(regional_values),
                        ),
                    ),
                )
            )

    return tuple(charts)


def _confidence_level(
    transaction_count: int,
    minimum_local_transactions: int,
) -> ConfidenceLevel:
    """Classify confidence from the deterministic sample-size policy."""

    if transaction_count < minimum_local_transactions:
        return ConfidenceLevel.LOW
    if transaction_count < 30:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.HIGH


def _percentage_change(start: float, end: float) -> float:
    """Calculate a rounded percentage change from non-zero values."""

    if start <= 0:
        raise ValueError("Percentage-change start value must be positive.")
    return round(((end - start) / start) * 100, 2)


def _normalise_series(values: Sequence[float]) -> tuple[float, ...]:
    """Normalize a numeric series to a base of 100."""

    if not values or values[0] <= 0:
        return ()
    return tuple(round((value / values[0]) * 100, 2) for value in values)


def _allowed_date_claims(evidence: EvidenceBundle) -> set[str]:
    """Build allowed textual date representations from evidence."""

    allowed: set[str] = set()

    dates = {
        value
        for window in evidence.source_windows
        for value in (
            window.start_date,
            window.end_date,
            window.latest_available_date,
        )
    }
    dates.update(point.period for point in evidence.monthly_local_metrics)

    for value in dates:
        allowed.update(
            {
                value.isoformat().casefold(),
                value.strftime("%Y-%m").casefold(),
                value.strftime("%B %Y").casefold(),
                value.strftime("%b %Y").casefold(),
            }
        )

    return allowed


def _collect_numbers(value: Any) -> tuple[float, ...]:
    """Recursively collect evidence numbers while excluding booleans."""

    numbers: list[float] = []

    if isinstance(value, bool):
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    if isinstance(value, dict):
        for nested in value.values():
            numbers.extend(_collect_numbers(nested))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            numbers.extend(_collect_numbers(nested))

    return tuple(numbers)


def _number_is_allowed(
    claim: str,
    allowed_numbers: Sequence[float],
) -> bool:
    """Allow supported percentages and conventionally rounded whole values.

    An unsigned percentage may express the magnitude of a signed evidence value,
    as in "a decline of 5%" for an evidence value of ``-5``. Explicitly signed
    claims must still match the evidence sign.
    """

    stripped = claim.replace("£", "").replace(",", "").strip()
    is_percentage = stripped.endswith("%")
    if is_percentage:
        stripped = stripped[:-1].strip()

    try:
        claimed_value = float(stripped)
    except ValueError:
        return False

    tolerance = 0.02 if is_percentage else 0.51
    return any(
        math.isclose(
            claimed_value,
            (
                abs(allowed)
                if is_percentage and claimed_value >= 0
                else allowed
            ),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for allowed in allowed_numbers
    )
