"""
Land Registry — data source usage examples.

Two endpoints are available, no authentication required:

  1. Price Paid SPARQL
     https://landregistry.data.gov.uk/landregistry/query
     Returns individual transaction records (price, date, address, property type).

  2. House Price Index REST
     https://landregistry.data.gov.uk/data/hpi/region/{region-name}.json
     Returns monthly regional index records (avg price, annual/monthly change).

Run this file to see example responses from both endpoints:
    python data_sources.py
"""

import json
import time
import urllib.request
import urllib.parse
from datetime import date, timedelta
from collections import defaultdict

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SPARQL_ENDPOINT = "https://landregistry.data.gov.uk/landregistry/query"
HPI_BASE        = "https://landregistry.data.gov.uk/data/hpi/region"

today     = date.today()
three_ago = today - timedelta(days=3 * 365)
FROM_DATE = three_ago.strftime("%Y-%m-%d")


def sparql_query(query: str, timeout: int = 60) -> dict:
    params = urllib.parse.urlencode({"query": query, "output": "json"})
    req = urllib.request.Request(
        f"{SPARQL_ENDPOINT}?{params}",
        headers={"Accept": "application/sparql-results+json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def hpi_get(region: str, params: dict | None = None, timeout: int = 20) -> dict:
    url = f"{HPI_BASE}/{region}.json"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(
        url, headers={"Accept": "application/json", "User-Agent": "lr-example/1.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------------------
# Example 1 — Price Paid transactions for a postcode district
#
# Namespace notes:
#   ppi:   http://landregistry.data.gov.uk/def/ppi/   — transaction predicates
#   common:http://landregistry.data.gov.uk/def/common/ — address predicates
#          (postcode, street, town live here, NOT under ppi:)
# ---------------------------------------------------------------------------

def example_price_paid_transactions(postcode_district: str = "GU1", limit: int = 10) -> list[dict]:
    """Return recent Price Paid transactions for a postcode district."""
    query = f"""
PREFIX ppi:    <http://landregistry.data.gov.uk/def/ppi/>
PREFIX common: <http://landregistry.data.gov.uk/def/common/>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

SELECT ?price ?date ?street ?postcode ?propertyType WHERE {{
  ?tx a ppi:TransactionRecord ;
      ppi:pricePaid       ?price ;
      ppi:transactionDate ?date ;
      ppi:propertyType    ?propertyType ;
      ppi:propertyAddress ?addr .
  ?addr common:postcode ?postcode ;
        common:street   ?street .
  FILTER(STRSTARTS(STR(?postcode), "{postcode_district} "))
  FILTER(?date >= "{FROM_DATE}"^^xsd:date)
}} LIMIT {limit}
"""
    data = sparql_query(query)
    rows = []
    for b in data["results"]["bindings"]:
        rows.append({
            "price":        int(b["price"]["value"]),
            "date":         b["date"]["value"],
            "street":       b["street"]["value"],
            "postcode":     b["postcode"]["value"],
            "propertyType": b["propertyType"]["value"].split("/")[-1],
        })
    return rows


# ---------------------------------------------------------------------------
# Example 2 — Top streets by average price (client-side aggregation)
#
# Avoid heavy GROUP BY / ORDER BY in SPARQL — the endpoint returns 503 under
# load for complex aggregations. Fetch flat rows and aggregate in Python.
# ---------------------------------------------------------------------------

def example_top_streets(postcode_district: str = "GU1", sample: int = 500, top_n: int = 5) -> list[dict]:
    """Return the top-N streets by average sale price for a postcode district."""
    query = f"""
PREFIX ppi:    <http://landregistry.data.gov.uk/def/ppi/>
PREFIX common: <http://landregistry.data.gov.uk/def/common/>
PREFIX xsd:    <http://www.w3.org/2001/XMLSchema#>

SELECT ?street ?price WHERE {{
  ?tx a ppi:TransactionRecord ;
      ppi:pricePaid       ?price ;
      ppi:transactionDate ?date ;
      ppi:propertyAddress ?addr .
  ?addr common:postcode ?pc ;
        common:street   ?street .
  FILTER(STRSTARTS(STR(?pc), "{postcode_district} "))
  FILTER(?date >= "{FROM_DATE}"^^xsd:date)
}} LIMIT {sample}
"""
    data = sparql_query(query)
    by_street: dict[str, list[float]] = defaultdict(list)
    for b in data["results"]["bindings"]:
        by_street[b["street"]["value"]].append(float(b["price"]["value"]))

    ranked = sorted(
        by_street.items(),
        key=lambda kv: sum(kv[1]) / len(kv[1]),
        reverse=True,
    )[:top_n]

    return [
        {"street": street, "avg_price": sum(prices) / len(prices), "transactions": len(prices)}
        for street, prices in ranked
    ]


# ---------------------------------------------------------------------------
# Example 3 — House Price Index for a region (last N months)
#
# The _sort=-refPeriod parameter returns newest first.
# refPeriod is a YYYY-MM string.
# NOTE: data currently only extends to 2016-03; the endpoint does not cover
#       the most recent years despite the date range in the exercise.
# ---------------------------------------------------------------------------

def example_hpi_region(region: str = "south-east", months: int = 36) -> list[dict]:
    """Return the most recent monthly HPI records for a region."""
    data = hpi_get(region, {"_pageSize": months, "_sort": "-refPeriod"})
    items = data["result"].get("items", [])
    rows = []
    for item in items:
        rows.append({
            "period":        item.get("refPeriod"),
            "avg_price":     item.get("averagePricesSASM"),
            "annual_change": item.get("annualChange"),
            "monthly_change": item.get("monthlyChange"),
        })
    return rows


# ---------------------------------------------------------------------------
# Run examples
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n── Price Paid: recent GU1 transactions ──────────────────────────")
    t0 = time.time()
    txns = example_price_paid_transactions("GU1", limit=5)
    print(f"   fetched in {round((time.time() - t0) * 1000)} ms")
    for t in txns:
        print(f"   £{t['price']:>10,}  {t['date']}  {t['street']:<35} {t['postcode']}  {t['propertyType']}")

    print("\n── Price Paid: top streets by avg price (GU1) ───────────────────")
    t0 = time.time()
    streets = example_top_streets("GU1", sample=500, top_n=5)
    print(f"   aggregated in {round((time.time() - t0) * 1000)} ms")
    for s in streets:
        print(f"   {s['street']:<40} £{s['avg_price']:>10,.0f}  ({s['transactions']} tx)")

    print("\n── HPI: South East — last 36 monthly records ────────────────────")
    t0 = time.time()
    hpi = example_hpi_region("south-east", months=36)
    print(f"   fetched in {round((time.time() - t0) * 1000)} ms  |  {len(hpi)} records")
    for h in hpi[:5]:
        print(f"   {h['period']}   avg £{h['avg_price']:>8,}   annual {h['annual_change']:>5}%   monthly {h['monthly_change']:>5}%")
    if len(hpi) > 5:
        print(f"   … {len(hpi) - 5} more records")
    print()
