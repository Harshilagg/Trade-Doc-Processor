# Evaluation set — how to fill in the ground truth

## The rule

**Ground truth is human-labelled only.**

Every expected value in `ground_truth.template.json` must come from a person reading
the document. Not from running the pipeline, not from the filename, not from a
previous run's output, not from anything generated.

This is not a style preference. If the expected values are populated from pipeline
output, the harness compares the pipeline against itself and scores 100% by
construction, no matter how wrong the extraction actually is. The measurement would
be worthless *and* would read as a perfect score, which is worse than having no
measurement at all.

The template ships with every value as an empty string for this reason. It was
generated without reading the contents of any document.

### The filenames are not labels

The files are named `Test-1_Approved(Nike).pdf`, `Test-2_Amendment(Nike).pdf` and
`Test-3_HumanReview(Apple) (1).pdf`. Those names appear to encode a decision and a
customer.

**Do not label from them.** They are someone's earlier expectation of what the
pipeline should do, which is exactly the thing under test. A filename saying
"Approved" is not evidence the document should be approved — that judgement has to
come from reading the document against the rules in
`ai-service/customer_rules.json`. If your reading disagrees with a filename, trust
your reading and record the disagreement in `notes`. That disagreement is itself a
finding.

## How to fill it in

1. Copy the template. Keep the template pristine so it can be regenerated:

   ```
   cp eval/ground_truth.template.json eval/ground_truth.json
   ```

   `ground_truth.json` is the file the harness reads.

2. Open each PDF and type what you actually see, field by field.

3. Fill in `_labelled_by` and `_labelled_date`. One person labelled this set; the
   report says so explicitly, and these fields are where that comes from.

4. Leave a field as `""` if the document genuinely does not contain it. An empty
   string means **"this document has no such value"** — a real, checkable
   expectation, not a skip. The harness treats a blank expected value as an
   assertion that the extractor should also find nothing there.

## The fields

Field names match `REQUIRED_FIELDS` in `ai-service/services/ai_service.py`.

| Field | What to record | Comparison at eval time |
|---|---|---|
| `consignee_name` | The receiving party's name, as printed | normalised |
| `hs_code` | The tariff code as printed, punctuation included | exact, after punctuation normalisation |
| `port_of_loading` | Origin port, as printed | normalised |
| `port_of_discharge` | Destination port, as printed | normalised |
| `incoterms` | The three-letter term, e.g. `FOB` | exact |
| `description_of_goods` | Free-text goods description | normalised |
| `gross_weight` | Weight with its unit, e.g. `1200 KG` | normalised |
| `invoice_number` | The invoice identifier as printed | exact |

"Normalised" means case, surrounding whitespace and punctuation are ignored when
comparing. "Exact" means the strings must match after that same whitespace/case
handling but with no further leniency — these are structured identifiers where a
single character difference is a genuine error.

For `hs_code`, the harness reuses the punctuation normalisation already implemented
in `ai-service/services/validator_agent.py` rather than reimplementing it, so
scoring cannot silently drift from validation. Record the code as printed —
`8471.30.00` and `84713000` normalise identically, so either is fine.

Type what is on the page. Do not correct a document's errors, expand its
abbreviations, or reformat its values — if the invoice says `SHANGHAI`, write
`SHANGHAI`. A typo in the source document is part of what the extractor has to
handle.

## Per-document fields

**`customer_id`** — which rule set in `ai-service/customer_rules.json` applies.
One of: `nike`, `adidas`, `zara`, `apple`, `maersk`, `generic`. This selects the
validation rules and the confidence threshold, so it changes the expected decision.
Determine it from the consignee on the document, not from the filename.

**`expected_decision`** — what the router should output, given the fields you
recorded and that customer's rules. Exactly one of:

| Value | When |
|---|---|
| `auto_approve` | every field matches its rule with adequate confidence |
| `human_review` | something is uncertain, but nothing contradicts a rule |
| `amendment_required` | at least one field contradicts a rule |

Values are from `ai-service/services/router_agent.py`. Precedence is: any mismatch
means `amendment_required`; otherwise any uncertainty means `human_review`;
otherwise `auto_approve`.

Deciding this means checking the fields you recorded against that customer's entry
in `customer_rules.json` — its `required_incoterms`, `allowed_ports_of_loading`,
`allowed_consignees` and `required_hs_code_prefix`. Do this from the rules file, by
hand. It is the slowest part of labelling and the part that matters most, because
decision accuracy is measured directly against it.

**`notes`** — anything ambiguous: a field you were unsure of, a value that was hard
to read, a disagreement with the filename, a document that seems to sit between two
decisions. Free text, ignored by scoring. Worth filling in — with a set this small,
the notes usually explain the failures better than the numbers do.

## What the set currently is

Three documents, measured for routing path by opening each one:

| File | Text layer | Path taken |
|---|---|---|
| `Test-1_Approved(Nike).pdf` | 272 chars, 74.7% alphabetic | digital (PyMuPDF) |
| `Test-2_Amendment(Nike).pdf` | 289 chars, 76.2% alphabetic | digital (PyMuPDF) |
| `Test-3_HumanReview(Apple) (1).pdf` | 18 chars, 1 embedded image | OCR (PaddleOCR) |

Two digital PDFs and one scan, so the OCR path has exactly one document behind it.

This set is small. Three documents cannot support percentages — the harness reports
counts with denominators (`2 of 3`), not rates, and the confidence-calibration
buckets will be very thin, with several likely empty. Treat every number it produces
as indicative. Adding documents, especially scans, is the single highest-value
change to the eval set.
