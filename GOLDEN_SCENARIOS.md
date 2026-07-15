# Golden Test Scenarios

Run the application with:

```bash
source .venv/bin/activate
streamlit run main.py
```

Live prices and transaction counts can change. Verify workflow behaviour rather than exact numerical values.

| ID | Scenario | Prompt / action | Expected result |
|---|---|---|---|
| G01 | Complete workflow | Analyse property price trends in GU1 over the last 3 years. Compare with the South East regional average. Identify the highest-value streets. Then prepare a one-paragraph research note and add it to my tracking sheet. | Deterministic interpretation. All five skills selected. Ten plan operations shown before retrieval. One-paragraph verified note. Workflow pauses for approval. Approval saves exactly one owner-scoped report. |
| G02 | Reject persistence | Run G01 with a new report name, then select **Reject report**. | Workflow pauses for approval. Rejection completes without saving. Audit trace records the rejection. |
| G03 | Explicitly prohibit writes | Analyse GU1 over three years, compare it with the South East, prepare a research note, but do not save it. | Persistence is forbidden. Approval and save operations are omitted. Evidence and note are displayed without a write. |
| G04 | Analysis only | Analyse prices in GU1 over five years, analysis only. | Only `local_property_trends` is selected. No note, verification, approval or save operations. |
| G05 | Regional comparison only | Analyse SW1A over two years. Compare it with London. Prepare a brief summary and do not save it. | Local, regional and note skills selected. No street-ranking or persistence skills. Note is one concise paragraph. |
| G06 | Street ranking only | Analyse M1 over three years, identify the highest-value streets, and prepare a detailed summary without persistence. | No HPI retrieval. Streets require at least three transactions. No approval or save operations. |
| G07 | Exact note length | Analyse GU1 over three years and prepare a three-paragraph report. Do not save it. | Detailed note requested with exactly three paragraphs. All numerical and date claims are verified. |
| G08 | LLM-assisted comparison | Analyse GU1 for three years against the wider South East and prepare a detailed summary. | Interpretation method is `LLM-assisted`. Regional comparison is selected. The plan still appears before external reads. |
| G09 | LLM-assisted street request | Analyse GU1 for three years and show me the priciest roads in a detailed write-up. | Interpretation method is `LLM-assisted`. Street-ranking skill is selected. Persistence is not added. |
| G10 | LLM-assisted persistence | Analyse GU1 for three years and record the result in my property tracker. | Interpretation method is `LLM-assisted`. Note and persistence skills are selected. Nothing is saved before approval. |
| G11 | Omit written note | Analyse GU1 for three years and leave out the written summary. | LLM-assisted interpretation resolves the omission. Analysis runs without drafting, verification or persistence. |
| G12 | Missing postcode | Compare property prices over four years with London. | Clarification is requested if the LLM cannot resolve a postcode from the request. No external reads or writes occur. |
| G13 | Conflicting persistence request | Analyse GU1 for three years, analysis only, then save it. | Validation explains that persistence requires a research note. Execution does not begin. |
| G14 | Unsupported subscription | Analyse GU1 for three years and subscribe to live updates. | Validation reports that subscriptions are outside the demonstration workflow. No autonomous loop or external write starts. |
| G15 | Non-overlapping periods | Analyse GU1 over the latest three years and compare it with the South East regional trend. Prepare a detailed note but do not save it. | Both source windows are displayed. If they do not overlap, like-for-like claims and the comparison chart are prohibited. |
| G16 | Sparse data | Analyse GU1 1AA over one year, identify the highest-value streets, and prepare a brief summary without persistence. | If fewer than ten transactions exist, confidence is low and percentage-change and street-ranking claims are suppressed. |
| G17 | Verification safety | Run any prompt requesting a note. | Verification follows drafting. At most one corrective redraft occurs. A second unsupported draft fails closed and never reaches approval. |
| G18 | Owner isolation | Run G01 as `golden-owner-a`, switch to `golden-owner-b` before approval, then switch back. | Owner B cannot approve Owner A’s report. After approval, only Owner A can list and open it. |
| G19 | External failure | Temporarily configure an unavailable model, then run: Analyse GU1 over three years and prepare a brief summary without persistence. | Run fails safely. The failed step and audit event are visible. No approval or save occurs. Restore `.env` afterwards. |
| G20 | Audit trace | Run any successful workflow. | Sequence numbers are contiguous, timestamps are timezone-aware, explanations are readable, and no credentials or hidden reasoning are stored. |

## Checks for every scenario

- The interpreted intent and interpretation method are visible.
- Selected skills and plan steps match the request.
- The plan is displayed before external data retrieval.
- Source windows, confidence and limitations are shown.
- Exact live values are not compared with hardcoded expectations.
- No write occurs without explicit approval.
- Failed or rejected runs do not create saved reports.
