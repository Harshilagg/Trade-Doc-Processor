# Findings

**These are recorded, not fixed — with one exception, finding 1, which was fixed
because the service could not run at all without it.** Every other finding's
behaviour was left exactly as found and pinned by a test, so that any future change
to it fails loudly rather than passing silently.

Ordered by severity. Measured on Python 3.13.5 / SQLite 3.53.1.

Findings 4, 5, 6 and 8 have tests asserting the **current** behaviour. Those tests
are not endorsements — they are tripwires. Fixing a finding means updating its test
in the same commit.

---

## 1. The pinned model is decommissioned — the service cannot run

**File:** `ai-service/config.py:42`

**Status: FIXED** in the same commit that added this entry. Recorded here because
it explains why `docs/ACCURACY.md`'s baseline exists at all, and what its numbers
actually describe.

**Measured:** `config.py:42` pinned `GROQ_MODEL = "llama-3.3-70b-versatile"`. Every
call to it returns:

```
404 - The model `llama-3.3-70b-versatile` does not exist or you do not have access to it
```

Querying the account's model list confirmed this is not an access or billing
problem — Groq has removed the model, and **no Llama chat model is available at
all.** The only `meta-llama` entries are `llama-prompt-guard-2-22m` and `-86m`,
which are prompt-safety classifiers, not chat models. Verified against two
different API keys, which returned identical lists.

Chat-capable models available, with JSON mode confirmed working (the extractor and
router both pass `response_format={"type": "json_object"}`): `openai/gpt-oss-120b`,
`openai/gpt-oss-20b`, `qwen/qwen3.8-27b`, `qwen/qwen3.6-27b`. `groq/compound-mini`
does not return valid JSON under that flag.

**Impact while broken:** total. All three agents read `Config.GROQ_MODEL`, so
extraction, router reasoning and the query agent were all failing. A `/process` call
exhausted all 3 extraction attempts, then returned null fields — and the failure was
**silent to the caller**: validation proceeded against nulls and the router emitted a
decision derived from an empty document rather than surfacing that the LLM never
ran. Two secondary problems visible in the same trace:

- The extractor retries a `404` twice (`MAX_LLM_RETRIES = 2`), treating a permanent
  error as transient and spending three round-trips to fail.
- Nothing in the pipeline distinguishes "the model returned nothing" from "the
  document contained nothing".

**The fix applied:** `config.py:42` now pins `openai/gpt-oss-120b`, the model
`docs/ACCURACY.md`'s baseline was measured on, so the committed configuration and
the committed numbers describe the same model. Nothing else was changed; no agent
logic was touched.

**Still worth addressing** (not done here): the model remains a hardcoded constant
with no environment fallback, so the next decommissioning requires a code change and
a redeploy. The retry-on-permanent-error and silent-null-extraction behaviours above
are untouched.

---

## 2. Reported confidence does not reflect OCR uncertainty

**Files:** `ai-service/services/ai_service.py` (extractor confidence),
`ai-service/customer_rules.json` (every `confidence_threshold`),
`ai-service/services/validator_agent.py:109-116` (the no-rule branch)

**Measured** over the 3-document eval set, one run per document. Every extracted
field bucketed by the confidence the extractor reported, against whether the value
was actually correct:

| Confidence bucket | Fields correct |
|---|---|
| 0.00 – 0.50 | 0 of 2 |
| 0.50 – 0.70 | no fields |
| 0.70 – 0.85 | 0 of 2 |
| 0.85 – 1.00 | 18 of 20 |

The top row looks like good calibration. It is not, and the aggregate is what hides
the problem. Split by extraction path:

- **Digital PDFs** (2 documents, 16 fields): every field returned at confidence
  **1.00**, and every field was correct. Confidence is perfectly calibrated here.
- **The scan** (1 document, 8 fields): only 2 of 8 correct — and the two errors
  sitting in the top bucket are both from this document:

| Field | Expected | Extracted | Confidence |
|---|---|---|---|
| `consignee_name` | `Apple Inc.` | `Aphle Inc.` | **0.95** |
| `invoice_number` | `APP-2026-001` | `APP-2O26-OO1` | **0.85** |

Both are OCR character confusions — `pp`→`ph`, and digit `0`→letter `O`. The
extractor reported 0.95 confidence on a corrupted company name. It cannot detect
that it is reading corrupted text, because the corruption is well-formed: `Aphle
Inc.` is a plausible-looking company name, and `APP-2O26-OO1` is a plausible-looking
invoice number.

**Consequence:** every `confidence_threshold` in `customer_rules.json` — nike 0.75,
adidas 0.75, apple 0.72, zara 0.70, maersk 0.65, generic 0.60 — gates on the
assumption that reported confidence predicts correctness. That assumption **holds on
the digital path and breaks on the OCR path**, which is precisely where extraction is
unreliable and where the gate is most needed. A corrupted value at 0.95 clears every
threshold in the file.

**Related gap — the no-rule branch launders corrupted values.** `APP-2O26-OO1` was
scored **`match`** by the validator. `apple` has no invoice-number rule, so the field
takes the no-rule branch at `validator_agent.py:109-116`, which returns `match` with
`expected: "any"` for any present value above the confidence threshold. A visibly
corrupted invoice number is therefore recorded as a successful match, not merely
unvalidated. Any field without a rule behaves this way, so the "match" count in a
validation summary conflates "checked and correct" with "present and unchecked".

**Observed downstream effect:** on the same document, `hs_code` was extracted as
`847I3O` (letter I, letter O) at confidence 0.80 — above apple's 0.72 threshold, so
it was rule-checked, failed the `8471` prefix, and became the single `mismatch`.
Under the precedence in finding 7, one mismatch outranks the four uncertains, so the
router returned `amendment_required` where the label was `human_review`. One OCR
character flip changed the pipeline's decision.

**A fix would involve:** deciding what confidence should mean when the text came from
OCR. The extractor sees only the text, not the fact that it was OCR'd, so it cannot
discount for that on its own — the extraction path is known to
`smart_extraction_pipeline` and could be passed through. Alternatives: propagate the
OCR engine's own per-token scores rather than asking the LLM to self-report; apply a
per-path threshold; or treat OCR-sourced fields as capped below the highest bucket.
Separately, the no-rule branch could return a distinct status such as `unchecked`
instead of `match`, so unvalidated fields stop counting as successes.

**Caveat:** 8 OCR fields from a single scanned document. This is indicative of a
mechanism, not a measured error rate. The digital-path result (16 of 16 at 1.00) is
equally thin.

### Second confirmation: model A/B tests are misleading on this metric

Measured while evaluating model routing for cost. Running **extraction** on
`openai/gpt-oss-20b` instead of `openai/gpt-oss-120b` produced, on the same set:

| | gpt-oss-120b | gpt-oss-20b |
|---|---|---|
| Fields correct | 18 of 24 | 18 of 24 |
| Decisions correct | 2 of 3 | **3 of 3** |
| Cost per document | $0.000838 | $0.000654 |

Read naively, the smaller model is more accurate and 21% cheaper. It is not. On the
scanned document the two models extracted **byte-identical values** — the same
`Aphle Inc.`, the same corrupted `847I3O`. Only the self-reported confidence
differed:

| Field | Extracted (both) | 120b confidence | 20b confidence | apple threshold |
|---|---|---|---|---|
| `hs_code` | `847I3O` | 0.80 | 0.60 | 0.72 |

At 0.80 the value clears the threshold, gets rule-checked, fails the `8471` prefix,
and becomes a `mismatch` → `amendment_required`. At 0.60 it falls below the
threshold, becomes `uncertain`, and the document routes to `human_review` — which
happens to match the hand-labelled expectation.

**The better decision score came from a lower confidence number on the same wrong
value, not from better extraction.** The pipeline's decision is being driven by a
self-reported figure that varies by model while correctness does not.

The practical consequence is a trap: anyone A/B-testing models on decision accuracy
over this set would conclude the smaller model is better and ship it, having
actually measured nothing about extraction quality. Any future model comparison
should be judged on extracted **values** first, with decision accuracy read only
alongside the confidences that produced it.

The 120b configuration is the one retained, because the value-level accuracy is
identical and the confidence it reports is at least closer to reflecting that the
field was legible enough to rule-check.

---

## 3. Every route is unauthenticated

**File:** `server/server.js:34-39`

```js
// ── Auth Middleware ──────────────────────────────────────────────────────────
const verifyToken = async (req, res, next) => {
    // Auth removed. Bypass token check and set default user.
    req.uid = 'local-user';
    next();
};
```

**Measured:** `verifyToken` performs no verification of any kind. It sets a constant
`req.uid = 'local-user'` and calls `next()` unconditionally. There is no JWT
library, no API-key check and no session lookup anywhere in the file. It is applied
to all 9 routes — `/documents` (43), `/upload` (54), `/trigger` (97),
`/documents/:docId/view` (175), `/shipments` (215), `/shipments/:shipmentId` (233),
`/query` (252), `/decisions` (272), `/stats` (284) — so the middleware is present
and wired everywhere, but enforces nothing.

The shape is the hazard: every route reads as protected at the call site. A reviewer
scanning route definitions sees `verifyToken` on each one and concludes auth is
handled.

**Impact:** Anyone who can reach the process can upload documents to S3, trigger
pipeline runs that spend Groq tokens, run queries, and read all extracted shipment
data. `req.uid` is a constant, so there is no per-user separation to fall back on.

**A fix would involve:** deciding what the trust boundary actually is. If the service
is genuinely local-only, the honest change is to delete `verifyToken` entirely and
document that the service must not be exposed — a no-op middleware named
`verifyToken` is worse than none, because it reads as protection. If it is reachable
by anyone else, real verification is needed in the middleware, plus a decision about
what `req.uid` means for data scoping, since nothing currently partitions data by
user.

---

## 4. The SQL guard does not stop stacked statements

**File:** `ai-service/utils/db_utils.py:413-415`

```python
sql_stripped = sql.strip().upper()
if not sql_stripped.startswith("SELECT"):
    raise ValueError("Only SELECT queries are permitted in the query agent.")
```

**Measured:** `"SELECT 1; DROP TABLE shipments"` **passes this guard** — it starts
with `SELECT`, so the check admits it and it reaches `conn.execute(sql, params)` on
line 418. What actually stops it is Python's `sqlite3` driver, which refuses
multi-statement strings:

```
sqlite3.ProgrammingError: You can only execute one statement at a time.
```

Confirmed against a throwaway database: the exception raised is
`sqlite3.ProgrammingError`, **not** the guard's `ValueError`, and the `shipments`
table survives. Same result for `"SELECT 1;DROP TABLE shipments;"` and for the
lowercase form.

So the SELECT-only claim holds in practice, but not for the reason the code
suggests. The protection is incidental to the driver API and sits one refactor away
from disappearing: route the same string through `executescript()`, or swap in a
driver that permits stacked statements, and the guard alone would let the `DROP`
through. This matters because the SQL is LLM-generated by the NL→SQL step in
`query_agent.py`, so the input to this guard is not fully controlled.

**A fix would involve:** rejecting `;` outside string literals before execution, or
parsing the statement rather than matching a prefix. Note that a naive `";" in sql`
check would reject legitimate queries containing a semicolon inside a quoted
literal, so the check needs to be literal-aware. Enforcing read-only at the
connection level — opening the query-agent connection with SQLite's read-only URI
mode — would be defence in depth that does not depend on parsing at all.

**Tests:** `ai-service/tests/test_sql_guard.py::TestStackedStatements` asserts the
`ProgrammingError` explicitly, including a test that the failure is *not* a
`ValueError`.

---

## 5. The guard rejects legitimate read-only queries

**File:** `ai-service/utils/db_utils.py:413-415` (same prefix check as finding 4)

**Measured:** both of these are safe, read-only, and refused with
`ValueError: Only SELECT queries are permitted in the query agent.`

| Query | Why it fails |
|---|---|
| `WITH t AS (SELECT * FROM shipments) SELECT COUNT(*) AS c FROM t` | starts with `WITH`, not `SELECT` |
| `-- count rows\nSELECT COUNT(*) AS c FROM shipments` | starts with a `--` comment |

Both forms are ordinary output from an LLM asked to write SQL — CTEs are a common
way to express aggregate questions, and leading explanatory comments are a very
common LLM habit. So this is a live failure mode of the query agent, not a
theoretical one.

**Impact:** availability, not security. The user gets a failed query for a question
the system could have answered.

**A fix would involve:** the same change as finding 4 — matching on the parsed
statement rather than the raw prefix would admit both of these and reject stacked
statements, resolving both findings together. Stripping leading comments and
whitespace before the check would handle the second row alone.

**Tests:** `ai-service/tests/test_sql_guard.py::TestGuardOverBlocksReadOnlyQueries`.

---

## 6. The fuzzy cutoff lifts unrelated company names over the threshold

**File:** `ai-service/services/validator_agent.py:143-147`

```python
best_similarity = max(
    _fuzzy_similarity(found_normalized, allowed)
    for allowed in allowed_normalized
)
if best_similarity >= 0.70:
```

`_fuzzy_similarity` (line 28) is `difflib.SequenceMatcher(None, a.upper(), b.upper()).ratio()`,
computed over the **whole string**. A score of `>= 0.70` is treated as a possible OCR
misread and downgraded to `uncertain` (human review) rather than `mismatch`
(amendment required).

**Measured** against nike's `allowed_consignees` from `customer_rules.json`, taking
the best ratio across the list:

| Candidate | Ratio | Outcome |
|---|---|---|
| `Aphle Imprts LL` | **0.7097** | `uncertain` — just above the cutoff |
| `Aphle Impo LLC` | **0.6667** | `mismatch` — just below the cutoff |
| `Zzzz Imports LLC` | **0.7500** | `uncertain` |

The first two bracket the cutoff tightly and behave sensibly — both are plausible
OCR corruptions of `Nike Imports LLC`.

The third is the problem. `Zzzz Imports LLC` shares no meaningful identity with
`Nike Imports LLC`, yet scores **0.7500** — higher than either genuine near-miss —
purely on the shared `" Imports LLC"` suffix. Because the ratio is computed over the
entire string, a common corporate suffix contributes most of the similarity. The
distinguishing part of the name is the part that is ignored.

**Impact:** a consignee that should raise `mismatch` and force an amendment is
instead reported as a probable scanning error and routed to human review. Since
finding 7 shows mismatch and uncertain lead to different decisions
(`amendment_required` vs `human_review`), this changes the pipeline's output, not
just its wording. The more standardised the customer's naming convention, the worse
it gets — an allow-list of `Nike Imports LLC` / `Nike Trading Company` /
`Nike Global Trading` makes any `<word> Trading` string score highly.

**A fix would involve:** comparing on the distinguishing portion rather than the
whole string — stripping common corporate suffixes (`LLC`, `Inc.`, `Ltd`, `Trading`,
`Company`) before scoring, or weighting the leading token. Note this cannot be
tuned away by raising the cutoff: `0.7500` sits *above* the genuine near-miss at
`0.7097`, so any threshold that rejects the unrelated name also rejects the real OCR
misread. The comparison itself has to change, not the number.

**Tests:** `ai-service/tests/test_validator_agent.py::TestConsigneeFuzzyMatching`,
including the measured ratios above.

---

## 7. "Nothing evaluated" and "no problems found" produce the same decision

**File:** `ai-service/services/router_agent.py:38-65`

```python
if match_count == total and total > 0:
    confidence = 0.97
    return DECISION_APPROVE, confidence

# Edge case: empty results → human review
return DECISION_REVIEW, 0.50
```

**Measured:** the auto-approve branch is guarded by `and total > 0`, so a completely
empty validation result (`total_fields = 0`) correctly avoids being read as "all
fields matched" and falls to `(human_review, 0.50)`. That guard is doing real work
and is right.

The consequence is that two very different situations arrive at the same output:

| Input | Decision | Meaning |
|---|---|---|
| `{mismatch: 0, uncertain: 0, match: 5, total: 8}` | `(human_review, 0.50)` | 5 matched, 3 unaccounted for |
| `{mismatch: 0, uncertain: 0, match: 0, total: 0}` | `(human_review, 0.50)` | nothing was evaluated at all |

Both reach the same fall-through. A reviewer receiving `human_review` at confidence
`0.50` cannot tell from the decision whether the document was mostly validated with
a few fields unaccounted for, or whether validation never ran — an upstream
extractor failure, for instance, which is a pipeline problem rather than a document
problem.

Routing an unknown state to human review is the safe direction, so this is a
diagnosability issue rather than a correctness one. The confidence value carries no
signal here either: `0.50` is a hardcoded constant for both rows, not a computed
score.

**A fix would involve:** distinguishing the empty case with its own decision or an
explicit reason string, so "validation did not run" is visibly different from "some
fields could not be accounted for". The deterministic decision itself need not
change — the two cases should just be separable downstream.

**Tests:** `ai-service/tests/test_router_agent.py::TestDecisionPrecedence` covers both
rows.

---

## 8. The two text-quality checks measure different lengths

**File:** `ai-service/services/extraction_service.py:9-14`

```python
if not text or len(text.strip()) < 80:
    return False

# Check for alphabetic character density (must be > 25% letters)
alpha_chars = sum(c.isalpha() for c in text)
if alpha_chars < (len(text) * 0.25):
    return False
```

**Measured:** the length check uses `len(text.strip())`, but the ratio check divides
by `len(text)` — **unstripped**. The same extracted text therefore changes extraction
path purely by gaining surrounding whitespace:

| Input | stripped len | raw len | alpha chars | Result |
|---|---|---|---|---|
| 25 letters + 75 digits | 100 | 100 | 25 | **digital fast path** |
| the same, plus 6 spaces | 100 | 106 | 25 | **OCR path** |

Identical content and an identical letter count. Six characters of padding push the
denominator from 100 to 106, dropping the ratio below `0.25` and sending the
document to PaddleOCR. Leading and trailing whitespace is exactly what varies
between PDF text layers, so this is reachable with real documents.

**Impact:** performance and cost, not correctness — the OCR path returns usable text,
just far more slowly than the fast path it should have taken. It also makes the
routing heuristic hard to reason about, since the threshold that decides the path
depends on invisible characters.

**Separately:** the comment on line 12 says *"must be > 25% letters"*, but the
implemented condition `alpha_chars < len * 0.25` admits **exactly** 25%. Measured:
24% → rejected, 25% → accepted, 26% → accepted. The README's `>= 25%` figure is the
correct description; the inline comment is the inaccurate one. Comment-only
discrepancy, no behavioural impact.

**A fix would involve:** computing `stripped = text.strip()` once and using it for
both checks. That is a one-line change, but it does shift the routing boundary for
documents near the threshold, so it should land together with a measurement of how
many documents in the eval set change path as a result — which is exactly what the
eval harness is being built to provide.

**Tests:** `ai-service/tests/test_extraction_routing.py::TestRatioDenominatorIsUnstripped`
and `::TestAlphabeticRatioBoundary`.
