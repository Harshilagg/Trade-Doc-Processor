# Extraction accuracy

Generated 2026-08-29 17:30 UTC by `eval/run_eval.py`.

## Read this first

**This set is 3 documents, hand-labelled by one person.** The numbers below are indicative, not measurements of general performance. Counts are reported with their denominators rather than as percentages, because a percentage over three documents implies a precision that does not exist here.

The confidence-calibration buckets are especially thin — several are likely empty or hold a single field. Read them as a sanity check on whether self-reported confidence means anything at all, not as a calibration curve.

Each document was run **3 times** (the extractor uses temperature 0.1, so output can vary). Accuracy is scored on run 1; the extra runs feed the stability check at the end.

Model used: **`openai/gpt-oss-120b`**.

## Field extraction accuracy

**18 of 24 field values correct** across 3 documents.

| Field | Correct | Comparison | Why |
|---|---|---|---|
| `consignee_name` | 2 of 3 | normalised | free text; case/spacing/punctuation vary by document |
| `hs_code` | 2 of 3 | exact (HS normalised) | structured code; punctuation normalised exactly as the validator does |
| `port_of_loading` | 3 of 3 | normalised | free text; e.g. trailing punctuation varies |
| `port_of_discharge` | 2 of 3 | normalised | free text; e.g. trailing punctuation varies |
| `incoterms` | 3 of 3 | exact | closed three-letter vocabulary; a near-miss is a different term |
| `description_of_goods` | 2 of 3 | normalised | free text; wording and casing vary |
| `gross_weight` | 2 of 3 | normalised | free text; unit spacing and punctuation vary |
| `invoice_number` | 2 of 3 | exact | identifier; one character off is a different invoice |

Normalised comparison ignores case, collapses whitespace and strips punctuation. Exact comparison ignores only surrounding whitespace and case. `hs_code` uses the validator's own normalisation (`.replace('.', '').replace(' ', '')`, `services/validator_agent.py:251`) so scoring cannot drift from validation.

Excluded from scoring: `container` — labelled in the ground truth but not in `REQUIRED_FIELDS` (`services/ai_service.py:29-38`), so the extractor has no concept of it and it could never match.

### Per document

| Document | Path | Fields correct |
|---|---|---|
| `Test-1_Approved(Nike).pdf` | digital | 8 of 8 |
| `Test-2_Amendment(Nike).pdf` | digital | 8 of 8 |
| `Test-3_HumanReview(Apple) (1).pdf` | ocr | 2 of 8 |

## Decision accuracy

**2 of 3 decisions correct** (run 1).

Rows are the hand-labelled expectation, columns the router's output.

| expected \ actual | `auto_approve` | `human_review` | `amendment_required` |
|---|---|---|---|
| `auto_approve` | 1 | 0 | 0 |
| `human_review` | 0 | 0 | 1 |
| `amendment_required` | 0 | 0 | 1 |

Ground-truth labels were written in prose and mapped onto the constants in `services/router_agent.py`: `amendment required` → `amendment_required`, `approved` → `auto_approve`, `auto approval` → `auto_approve`, `auto approve` → `auto_approve`, `human review` → `human_review`. An unrecognised label aborts the run rather than scoring as a miss.

## Confidence calibration

Every extracted field bucketed by the confidence the extractor reported for it, against whether that field was actually correct. This tests whether self-reported confidence predicts correctness at all — which every `confidence_threshold` in `customer_rules.json` assumes.

| Confidence bucket | Fields correct |
|---|---|
| 0.00 – 0.50 | 0 of 2 |
| 0.50 – 0.70 | no fields |
| 0.70 – 0.85 | 0 of 2 |
| 0.85 – 1.00 | 18 of 20 |

3 of 4 buckets contain any fields. With this few documents the buckets cannot show a calibration trend; they show only whether confidence and correctness are wildly inconsistent.

## Latency

Wall-clock per run. `extract` is text extraction only (the PyMuPDF or PaddleOCR step); `total` includes the extractor LLM call, validation and the router LLM call.

| Document | Path | extract (s) | total (s) |
|---|---|---|---|
| `Test-1_Approved(Nike).pdf` | digital | 0.01, 0.01, 0.01 | 3.80, 3.32, 3.42 |
| `Test-2_Amendment(Nike).pdf` | digital | 0.01, 0.01, 0.01 | 4.75, 25.54, 17.01 |
| `Test-3_HumanReview(Apple) (1).pdf` | ocr | 13.95, 10.47, 15.74 | 21.99, 24.17, 21.09 |

- **digital**: 6 runs over 2 document(s), text extraction 0.01–0.01s
- **ocr**: 3 runs over 1 document(s), text extraction 10.47–15.74s

The two paths are backed by very few documents (2 digital, 1 OCR), so this split is a single observation per path rather than a comparison.

## Run-to-run stability

No field changed across 3 runs of each document. That is not proof of determinism — temperature is 0.1, not 0 — only that no variation surfaced in this many runs.

## Appendix — every field, expected vs extracted (run 1)

With a set this small the individual misses say more than the totals.

**`Test-1_Approved(Nike).pdf`** (digital path)

| Field | Expected | Extracted | Confidence | |
|---|---|---|---|---|
| `consignee_name` | Nike India Pvt Ltd | Nike India Pvt Ltd | 1.00 | ok |
| `hs_code` | 847130 | 847130 | 1.00 | ok |
| `port_of_loading` | Shanghai | Shanghai | 1.00 | ok |
| `port_of_discharge` | Mumbai | Mumbai | 1.00 | ok |
| `incoterms` | FOB | FOB | 1.00 | ok |
| `description_of_goods` | Laptop Computers | Laptop Computers | 1.00 | ok |
| `gross_weight` | 1250 KG | 1250 KG | 1.00 | ok |
| `invoice_number` | INV-2026-001 | INV-2026-001 | 1.00 | ok |

**`Test-2_Amendment(Nike).pdf`** (digital path)

| Field | Expected | Extracted | Confidence | |
|---|---|---|---|---|
| `consignee_name` | Nike Asia Imports | Nike Asia Imports | 1.00 | ok |
| `hs_code` | 847131 | 847131 | 1.00 | ok |
| `port_of_loading` | Shenzhen | Shenzhen | 1.00 | ok |
| `port_of_discharge` | Mumbai | Mumbai | 1.00 | ok |
| `incoterms` | CIF | CIF | 1.00 | ok |
| `description_of_goods` | Laptop Computers | Laptop Computers | 1.00 | ok |
| `gross_weight` | 1180 KG | 1180 KG | 1.00 | ok |
| `invoice_number` | INV-2026-002 | INV-2026-002 | 1.00 | ok |

**`Test-3_HumanReview(Apple) (1).pdf`** (ocr path)

| Field | Expected | Extracted | Confidence | |
|---|---|---|---|---|
| `consignee_name` | Apple Inc. | Aphle Inc. | 0.95 | **miss** |
| `hs_code` | 847130 | 847I3O | 0.80 | **miss** |
| `port_of_loading` | Shenzhen | Shenzhen | 0.95 | ok |
| `port_of_discharge` | Los Angeles | — | 0.00 | **miss** |
| `incoterms` | DAP | DAP | 0.98 | ok |
| `description_of_goods` | Laptop Computers | — | 0.00 | **miss** |
| `gross_weight` | 1850 KG | 7185O KG | 0.75 | **miss** |
| `invoice_number` | APP-2026-001 | APP-2O26-OO1 | 0.90 | **miss** |

## Method

- Documents are read from `test-documents/` on local disk; S3 is not used.
- Each document runs the real pipeline: `smart_extraction_pipeline` → `extract_shipment_fields` → `validate_shipment` → `route_decision`.
- Ground truth is `eval/ground_truth.json`, hand-labelled. This harness never writes to it.
- Extraction path is determined by re-running the same check `smart_extraction_pipeline` uses, so attribution matches the pipeline's own routing.

Reproduce with `python eval/run_eval.py --runs 3`.
