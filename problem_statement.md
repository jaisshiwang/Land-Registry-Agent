***

# Take-Home Exercise — AI Engineer
**PE Limited · Petex**

**Time allocation:** 3–4 hours
**Submission:** GitHub repo (private or public). Please share with `dminskiy` if private.

**Please treat this task description as confidential and DO NOT distribute it.**

***

## Context

We are building an AI orchestration platform that connects oil and gas engineers to Petex simulation tools (PROSPER, GAP, MBAL, RESOLVE, and others) and the wider organisation toolset in a single AI-powered platform. Engineers describe what they want in natural language; the platform plans, executes tool calls, interprets results, and, only after explicit approval, performs write actions against live systems. Read operations do not require approval.

Before integrating with Petex simulation models, we validate architecture decisions against readily available public data. This exercise uses **UK Land Registry house price data** as a structural proxy:

| Land Registry concept | O&G analogue |
|---|---|
| Transaction records per postcode | Well production timeseries per field zone |
| Price trend over time | Production decline / pressure depletion |
| Region-level index | Reservoir-level aggregated performance |
| Write research note to tracking sheet | Write interpreted result to DOF model |

***

## What You Are Building

An agentic harness that:
1. Retrieves and analyses UK property price data for a given area
2. Reasons over the data to produce a short research note
3. **Only after user approval**, writes that note somewhere persistent

The target user prompt is (we will test other requests too):

> *"Analyse property price trends in GU1 over the last 3 years. Compare with the South East regional average. Identify the highest-value streets. Then prepare a one-paragraph research note and add it to my tracking sheet."*

Note, the data sources are free and require no authentication:
- Dataset Description: https://landregistry.data.gov.uk/
- Price Paid data (SPARQL): https://landregistry.data.gov.uk/landregistry/query
- House Price Index (REST): https://landregistry.data.gov.uk/data/hpi/region/{region-name}.json

A working starter script demonstrating both endpoints is provided in `data_sources.py`.

***


## Deliverables

### 1 — The Agent
Write an agent that fulfils at least the user prompt above. It should:
- **Plan before executing** — produce a structured plan before issuing any tool calls
- **Gate write actions** — the note must not be written until the user explicitly approves it. Reading is fine. Ensure there is an audit trail of decisions taken for compliance.
- **Return structured output** — at minimum:
    - the research note;
    - chart data (rendered or JSON);
    - and a trace of what the agent did and in what order. This trace is what a non-technical engineer would read to understand what the agent did and why, so design it for that reader.

### 2 — Data Access Layer
Build something that retrieves the data the agent needs. How you expose that (as a direct API wrapper, as an internal service, or otherwise) is **your decision**. Justify it.

If sparse data is encountered (e.g. fewer than 10 transactions for a postcode), the system should handle this gracefully rather than failing or hallucinating.

### 3 — README
Your README is a first-class deliverable. Cover:
- The architectural decisions you made and why
- What you deliberately left out or simplified and why
- What you would invest in next with more time

***

## What We Are Evaluating

We are looking for evidence of how you think at the architectural level. We are especially interested in your decision-making and prioritisation rationale.

| Signal | What we're listening for |
|---|---|
| **Integration strategy** | How extendable is the solution? What does that choice cost and gain you in the context of a production platform? |
| **Architectural decisions** | How did you separate concerns? How do you handle tool schema design, agent-to-tool contracts, and write-gating? |
| **Deliberate trade-offs** | What did you cut, mock, or defer — and can you articulate the reasoning clearly? |
| **Operational instincts** | Sparse data, slow endpoints, hallucinated tool calls — do you handle failure modes or ignore them? |
| **Code quality** | Typed, structured, testable. |

***

## Constraints

- **Python** for the backend; frameworks of your choice
- **LLM provider** — we will provide an OpenAI key. You are welcome to use another provider if you prefer, but you will need to supply your own key.

***

## Genuinely Optional Bonuses

- Render charts in a minimal frontend component
- Use of skills to control the execution patterns
- Advanced use of MCPs, eg `subscribe_resource` pattern that triggers re-fetch on resource URI change

***

## A Note on Scope

Three to four hours will not be enough to build everything above to production quality — and we don't expect it to be. What we are evaluating is the quality of your judgment about what to build, what to defer, and how to communicate those choices. The README explanation of your design decisions often tells us more than the code itself. However, we do hold high standards.

***

## Known API issues
These are real behaviours of the live endpoints. Handling them gracefully is part of the exercise.

### Price Paid SPARQL endpoint
* Slow responses. Simple filtered queries over GU1 take 10–20 seconds. Aggregation queries take longer. Caching responses locally (even in-memory for the session) is expected and counts as good instincts.
* 503 under load. Complex `GROUP BY` / `ORDER BY` / `AVG` queries over large result sets intermittently return HTTP 503. The recommended pattern is to fetch flat rows with a `LIMIT` and aggregate in Python rather than pushing aggregation into SPARQL. See `data_sources.py` for an example.
* *Address fields use a different namespace. `postcode`, `street`, and `town` are under `common:` (`http://landregistry.data.gov.uk/def/common/`), not under `ppi:`. Queries that use `ppi:postcode` or `ppi:street` will return no results.
* *Property type values are URIs. `ppi:propertyType` returns a full URI such as `http://landregistry.data.gov.uk/def/ppi/detachedType`. Extract the label with `.split("/")[-1]`.

### House Price Index REST endpoint
* The URL in older documentation is wrong. The path `/api/1/slice/linked-hpi.json` returns 404. The correct pattern is `/data/hpi/region/{region-name}.json` (e.g. `south-east`, `london`, `england`).
* Data does not extend to the present day. The most recent available `refPeriod` is currently `2016-03`. Querying for "the last 3 years" will return data from that period, not the current date. Design your agent's reasoning to use the most recent available window rather than assuming recency.
* Pagination is available but optional for a 3-year window. A single page with `_pageSize=36` covers 36 months. A `next` key will be present in the response; implement page-walking if you need the full historical dataset.
