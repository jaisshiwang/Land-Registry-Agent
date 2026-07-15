"""Typed access to HM Land Registry Price Paid and regional HPI data."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar
from urllib.parse import urljoin

import httpx

from land_registry_agent.config import Settings
from land_registry_agent.models import HPIRecord, PropertyType, Transaction

JsonObject = dict[str, Any]
RecordT = TypeVar("RecordT")


class DataSourceError(RuntimeError):
    """Raised when an external data source cannot return usable data."""


class PaginationLimitError(DataSourceError):
    """Raised when a bounded pagination limit is exhausted."""


@dataclass(frozen=True)
class FetchResult(Generic[RecordT]):
    """Normalised records and safe provenance from one gateway operation."""

    records: tuple[RecordT, ...]
    source_url: str
    artifact_keys: tuple[str, ...]


class PropertyDataGateway(Protocol):
    """Interface consumed by the deterministic workflow."""

    def fetch_price_paid_transactions(
        self,
        postcode: str,
        requested_years: int,
    ) -> FetchResult[Transaction]:
        """Fetch flat Price Paid rows for a postcode or district."""

    def fetch_regional_hpi(
        self,
        region: str,
        requested_years: int,
    ) -> FetchResult[HPIRecord]:
        """Fetch the latest available regional HPI window."""


class DiskJsonCache:
    """Small TTL cache for successful HTTP JSON responses."""

    def __init__(self, directory: Path, ttl_seconds: int) -> None:
        self._directory = directory
        self._ttl_seconds = ttl_seconds

    def key_for(
        self,
        url: str,
        params: Mapping[str, str | int] | None,
    ) -> str:
        """Build a stable key without storing credentials or hidden state."""

        canonical_request = json.dumps(
            {"url": url, "params": dict(params or {})},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()

    def get(self, key: str) -> JsonObject | None:
        """Return a fresh cached response, ignoring missing or corrupt entries."""

        if self._ttl_seconds == 0:
            return None

        path = self._directory / f"{key}.json"
        try:
            age_seconds = time.time() - path.stat().st_mtime
            if age_seconds > self._ttl_seconds:
                return None

            payload = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return None

        return payload if isinstance(payload, dict) else None

    def put(self, key: str, payload: JsonObject) -> None:
        """Atomically cache a successful response."""

        if self._ttl_seconds == 0:
            return

        self._directory.mkdir(parents=True, exist_ok=True)
        destination = self._directory / f"{key}.json"
        temporary = self._directory / f"{key}.tmp"
        temporary.write_text(
            json.dumps(payload, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(destination)


class LandRegistryGateway:
    """HTTP gateway using the exercise's documented endpoint behaviour."""

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.Client | None = None,
        cache: DiskJsonCache | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=settings.http_timeout_seconds,
            follow_redirects=True,
            headers={
                "Accept": "application/json",
                "User-Agent": "land-registry-agent-demo/1.0",
            },
        )
        self._cache = cache or DiskJsonCache(
            settings.cache_directory,
            settings.cache_ttl_seconds,
        )

    def close(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_client:
            self._client.close()

    def fetch_price_paid_transactions(
        self,
        postcode: str,
        requested_years: int,
    ) -> FetchResult[Transaction]:
        """Fetch bounded, flat transaction rows for local aggregation."""

        normalised_postcode = _normalise_postcode(postcode)
        end_date = date.today()
        start_date = _years_before(end_date, requested_years)

        transactions: dict[str, Transaction] = {}
        artifact_keys: list[str] = []

        for page_number in range(self._settings.query_max_pages):
            query = _build_price_paid_query(
                postcode=normalised_postcode,
                start_date=start_date,
                end_date=end_date,
                limit=self._settings.query_page_size,
                offset=page_number * self._settings.query_page_size,
            )
            payload, artifact_key = self._get_json(
                str(self._settings.land_registry_sparql_url),
                params={"query": query, "output": "json"},
            )
            artifact_keys.append(artifact_key)
            bindings = _sparql_bindings(payload)

            for binding in bindings:
                transaction = _parse_transaction(binding)
                transactions[transaction.transaction_id] = transaction

            if len(bindings) < self._settings.query_page_size:
                break
        else:
            raise PaginationLimitError(
                "Price Paid results exceeded the configured pagination limit."
            )

        ordered = tuple(
            sorted(
                transactions.values(),
                key=lambda item: (item.transfer_date, item.transaction_id),
            )
        )
        return FetchResult(
            records=ordered,
            source_url=str(self._settings.land_registry_sparql_url),
            artifact_keys=tuple(artifact_keys),
        )

    def fetch_regional_hpi(
        self,
        region: str,
        requested_years: int,
    ) -> FetchResult[HPIRecord]:
        """Fetch the newest available HPI records, even when historically stale."""

        region_slug = _region_slug(region)
        requested_months = requested_years * 12
        source_url = (
            f"{str(self._settings.hpi_region_base_url).rstrip('/')}"
            f"/{region_slug}.json"
        )

        records_by_period: dict[date, HPIRecord] = {}
        artifact_keys: list[str] = []
        next_url: str | None = source_url
        params: Mapping[str, str | int] | None = {
            "_pageSize": requested_months,
            "_sort": "-refPeriod",
        }

        for _ in range(self._settings.query_max_pages):
            if next_url is None or len(records_by_period) >= requested_months:
                break

            payload, artifact_key = self._get_json(next_url, params=params)
            artifact_keys.append(artifact_key)
            result = payload.get("result")

            if not isinstance(result, dict):
                raise DataSourceError("HPI response is missing its result object.")

            items = result.get("items", [])
            if not isinstance(items, list):
                raise DataSourceError("HPI response items are not a list.")

            for item in items:
                if not isinstance(item, dict):
                    raise DataSourceError("HPI response contains an invalid item.")
                record = _parse_hpi_record(item, region)
                records_by_period[record.period] = record

            raw_next = result.get("next") or payload.get("next")
            next_url = (
                urljoin(next_url, raw_next)
                if isinstance(raw_next, str) and raw_next
                else None
            )
            params = None
        else:
            raise PaginationLimitError(
                "HPI results exceeded the configured pagination limit."
            )

        newest_first = sorted(
            records_by_period.values(),
            key=lambda item: item.period,
            reverse=True,
        )[:requested_months]

        return FetchResult(
            records=tuple(reversed(newest_first)),
            source_url=source_url,
            artifact_keys=tuple(artifact_keys),
        )

    def _get_json(
        self,
        url: str,
        *,
        params: Mapping[str, str | int] | None = None,
    ) -> tuple[JsonObject, str]:
        """GET JSON with caching and bounded retries for transient failures."""

        artifact_key = self._cache.key_for(url, params)
        cached = self._cache.get(artifact_key)
        if cached is not None:
            return cached, artifact_key

        last_error: Exception | None = None

        for attempt in range(self._settings.http_max_retries + 1):
            try:
                response = self._client.get(url, params=params)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt < self._settings.http_max_retries:
                    self._sleep(2**attempt)
                    continue
                break

            if response.status_code == 429 or response.status_code >= 500:
                last_error = DataSourceError(
                    f"Data source returned HTTP {response.status_code}."
                )
                if attempt < self._settings.http_max_retries:
                    self._sleep(_retry_delay(response, attempt))
                    continue
                break

            try:
                response.raise_for_status()
                payload = response.json()
            except httpx.HTTPStatusError as exc:
                raise DataSourceError(
                    f"Data source returned HTTP {response.status_code}."
                ) from exc
            except ValueError as exc:
                raise DataSourceError(
                    "Data source returned malformed JSON."
                ) from exc

            if not isinstance(payload, dict):
                raise DataSourceError("Data source returned an unexpected JSON shape.")

            self._cache.put(artifact_key, payload)
            return payload, artifact_key

        raise DataSourceError(
            "Data source remained unavailable after bounded retries."
        ) from last_error


def _build_price_paid_query(
    *,
    postcode: str,
    start_date: date,
    end_date: date,
    limit: int,
    offset: int,
) -> str:
    """Build a flat-row SPARQL query with no server-side aggregation."""

    if " " in postcode:
        postcode_filter = f'UCASE(STR(?postcode)) = "{postcode}"'
    else:
        postcode_filter = (
            f'STRSTARTS(UCASE(STR(?postcode)), "{postcode} ")'
        )

    return f"""
PREFIX ppi:    <http://landregistry.data.gov.uk/def/ppi/>
PREFIX common: <http://landregistry.data.gov.uk/def/common/>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

SELECT ?transactionId ?price ?date ?street ?postcode ?propertyType WHERE {{
  ?tx a ppi:TransactionRecord ;
      ppi:transactionId   ?transactionId ;
      ppi:pricePaid       ?price ;
      ppi:transactionDate ?date ;
      ppi:propertyAddress ?address .

  ?address common:postcode ?postcode .

  OPTIONAL {{ ?address common:street ?street . }}
  OPTIONAL {{ ?tx ppi:propertyType ?propertyType . }}

  FILTER({postcode_filter})
  FILTER(?date >= "{start_date.isoformat()}"^^xsd:date)
  FILTER(?date <= "{end_date.isoformat()}"^^xsd:date)
}}
ORDER BY ?date ?transactionId
LIMIT {limit}
OFFSET {offset}
""".strip()


def _sparql_bindings(payload: JsonObject) -> list[JsonObject]:
    """Extract and validate SPARQL result bindings."""

    try:
        bindings = payload["results"]["bindings"]
    except (KeyError, TypeError) as exc:
        raise DataSourceError(
            "SPARQL response is missing result bindings."
        ) from exc

    if not isinstance(bindings, list) or not all(
        isinstance(item, dict) for item in bindings
    ):
        raise DataSourceError("SPARQL bindings have an unexpected shape.")

    return bindings


def _parse_transaction(binding: JsonObject) -> Transaction:
    """Normalise one SPARQL binding into the domain model."""

    try:
        return Transaction(
            transaction_id=_binding_value(binding, "transactionId"),
            transfer_date=date.fromisoformat(
                _binding_value(binding, "date")[:10]
            ),
            price_gbp=int(_binding_value(binding, "price")),
            postcode=_binding_value(binding, "postcode").upper(),
            property_type=_normalise_property_type(
                _optional_binding_value(binding, "propertyType")
            ),
            street=_optional_binding_value(binding, "street"),
        )
    except (TypeError, ValueError) as exc:
        raise DataSourceError(
            "SPARQL returned an invalid transaction row."
        ) from exc


def _parse_hpi_record(item: JsonObject, region: str) -> HPIRecord:
    """Normalise one item from the supplied HPI REST response."""

    try:
        period_text = str(item["refPeriod"])
        average_price = _required_float(item, "averagePricesSASM")
        return HPIRecord(
            period=date.fromisoformat(f"{period_text}-01"),
            region=region,
            average_price_gbp=average_price,
            annual_change_percentage=_optional_float(item.get("annualChange")),
            monthly_change_percentage=_optional_float(item.get("monthlyChange")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataSourceError("HPI returned an invalid monthly record.") from exc


def _binding_value(binding: JsonObject, name: str) -> str:
    """Return a required scalar from a SPARQL binding."""

    value = binding[name]["value"]
    if not isinstance(value, str) or not value:
        raise ValueError(f"Binding {name!r} is empty.")
    return value


def _optional_binding_value(
    binding: JsonObject,
    name: str,
) -> str | None:
    """Return an optional scalar from a SPARQL binding."""

    item = binding.get(name)
    if not isinstance(item, dict):
        return None
    value = item.get("value")
    return value if isinstance(value, str) and value else None


def _normalise_property_type(value: str | None) -> PropertyType:
    """Convert the API's property-type URI into the internal label."""

    if value is None:
        return "unknown"

    suffix = value.split("/")[-1].lower()
    property_types: Mapping[str, PropertyType] = {
        "detachedtype": "detached",
        "detached": "detached",
        "semidetachedtype": "semi_detached",
        "semi-detached": "semi_detached",
        "terracedtype": "terraced",
        "terraced": "terraced",
        "flatmaisonettetype": "flat_maisonette",
        "flat-maisonette": "flat_maisonette",
        "otherpropertytype": "other",
        "other": "other",
    }
    return property_types.get(suffix, "unknown")


def _normalise_postcode(postcode: str) -> str:
    """Normalise and defensively validate a postcode query value."""

    normalised = " ".join(postcode.upper().split())
    if not re.fullmatch(r"[A-Z0-9 ]{2,10}", normalised):
        raise ValueError("Postcode contains unsupported characters.")
    return normalised


def _region_slug(region: str) -> str:
    """Convert a policy-validated region into the REST path form."""

    slug = re.sub(r"[^a-z]+", "-", region.lower()).strip("-")
    if not slug:
        raise ValueError("Region must not be blank.")
    return slug


def _required_float(item: JsonObject, name: str) -> float:
    """Parse a required numeric API value."""

    value = item[name]
    if value is None:
        raise ValueError(f"Field {name!r} is missing.")
    return float(str(value).replace(",", ""))


def _optional_float(value: object) -> float | None:
    """Parse an optional numeric API value."""

    if value is None or value == "":
        return None
    return float(str(value).replace(",", ""))


def _retry_delay(response: httpx.Response, attempt: int) -> float:
    """Use a numeric Retry-After value or bounded exponential backoff."""

    retry_after = response.headers.get("Retry-After")
    if retry_after is not None:
        try:
            return min(float(retry_after), 30.0)
        except ValueError:
            pass
    return min(float(2**attempt), 30.0)


def _years_before(value: date, years: int) -> date:
    """Subtract calendar years while handling a leap-day boundary."""

    try:
        return value.replace(year=value.year - years)
    except ValueError:
        return value.replace(year=value.year - years, day=28)
