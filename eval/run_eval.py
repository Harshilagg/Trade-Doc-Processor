"""
Evaluation harness — measures the pipeline against hand-labelled ground truth.

Runs each document through the real pipeline (extraction -> validation -> routing)
against local files, bypassing S3. Ground truth is never written by this script.

Usage:
    cd ai-service && ../eval/run_eval.py          # or:
    python eval/run_eval.py [--runs N] [--out docs/ACCURACY.md]

Reports counts with denominators, not percentages: the document set is far too
small for a rate to mean anything.
"""
import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
AI_SERVICE = os.path.join(REPO_ROOT, "ai-service")
INVOKED_FROM = os.getcwd()
sys.path.insert(0, AI_SERVICE)

# config.py resolves .env, customer_rules.json and the SQLite path relative to the
# process cwd, and constructs the Groq client at import time. So the harness must
# be in ai-service/ before these imports, exactly as the service is when it runs.
os.chdir(AI_SERVICE)

from services.ai_service import REQUIRED_FIELDS, extract_shipment_fields  # noqa: E402
from services.extraction_service import (  # noqa: E402
    extract_digital_text,
    smart_extraction_pipeline,
    validate_text_quality,
)
from services.router_agent import route_decision  # noqa: E402
from utils import llm_metrics  # noqa: E402
from services.validator_agent import validate_shipment  # noqa: E402

# ── Decision vocabulary ───────────────────────────────────────────────────────
# The ground truth is written in prose; the router emits the constants defined in
# services/router_agent.py. This map is the only bridge between them, and an
# unrecognised label is a hard error rather than a silent miss.
DECISIONS = ["auto_approve", "human_review", "amendment_required"]
DECISION_ALIASES = {
    "auto approval": "auto_approve",
    "auto approve": "auto_approve",
    "approved": "auto_approve",
    "human review": "human_review",
    "amendment required": "amendment_required",
}

CONFIDENCE_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]

# Which model the run actually used vs. what config.py pins. Populated in main().
MODEL_NOTE = {"configured": None, "used": None, "reasoning": None}

# Fields compared exactly: structured identifiers where one character is an error.
EXACT_FIELDS = {"hs_code", "incoterms", "invoice_number"}


def canonical_decision(label):
    """Map a ground-truth decision label onto a router constant, or fail loudly."""
    key = re.sub(r"[\s_-]+", " ", str(label).strip().lower())
    if key.replace(" ", "_") in DECISIONS:
        return key.replace(" ", "_")
    if key in DECISION_ALIASES:
        return DECISION_ALIASES[key]
    raise ValueError(
        f"Unrecognised expected_decision {label!r}. "
        f"Expected one of {DECISIONS} or a known alias {sorted(DECISION_ALIASES)}."
    )


def normalise_text(value):
    """Case, whitespace and punctuation insensitive comparison for free text."""
    if value is None:
        return ""
    text = str(value).lower().strip()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def normalise_hs_code(value):
    """
    HS-code normalisation, matching services/validator_agent.py:251:
        str(...).replace(".", "").replace(" ", "")
    Kept identical so eval scoring cannot disagree with validation. The validator
    applies this inline rather than exposing a helper, so it is mirrored here
    rather than imported; test_eval_comparators.py asserts the two agree.
    """
    if value is None:
        return ""
    return str(value).replace(".", "").replace(" ", "").upper()


def fields_match(field_name, expected, actual):
    """Returns (is_match, comparison_kind)."""
    if field_name == "hs_code":
        return normalise_hs_code(expected) == normalise_hs_code(actual), "exact/hs"
    if field_name in EXACT_FIELDS:
        e = str(expected or "").strip().upper()
        a = str(actual or "").strip().upper()
        return e == a, "exact"
    return normalise_text(expected) == normalise_text(actual), "normalised"


def bucket_of(confidence):
    for low, high in CONFIDENCE_BUCKETS:
        if low <= confidence < high or (high == 1.0 and confidence >= high):
            return (low, high)
    return None


def detect_path(file_path):
    """
    Which extraction path this document takes. Mirrors the decision in
    services/extraction_service.py:smart_extraction_pipeline without altering it,
    so the harness can attribute latency without the pipeline reporting it.
    """
    if any(file_path.lower().endswith(e) for e in (".png", ".jpg", ".jpeg")):
        return "ocr"
    return "digital" if validate_text_quality(extract_digital_text(file_path)) else "ocr"


def run_document(file_path, customer_id):
    """One full pipeline pass over one local document."""
    llm_metrics.reset()
    t0 = time.perf_counter()
    raw_text = smart_extraction_pipeline(file_path)
    t_extract = time.perf_counter() - t0

    t1 = time.perf_counter()
    extraction = extract_shipment_fields(raw_text)
    t_llm = time.perf_counter() - t1

    fields = extraction.get("extracted_fields", {}) or {}
    validation = validate_shipment(fields, customer_id)
    routing = route_decision(validation, customer_name=validation.get("customer_name"))

    return {
        "fields": fields,
        "decision": routing.get("decision"),
        "validation": validation,
        "retry_count": extraction.get("retry_count", 0),
        "extraction_seconds": round(t_extract, 3),
        "llm_seconds": round(t_llm, 3),
        "total_seconds": round(time.perf_counter() - t0, 3),
        "raw_text_length": len(raw_text or ""),
        "llm_calls": llm_metrics.get_calls(),
    }


def value_of(fields, name):
    entry = fields.get(name)
    if isinstance(entry, dict):
        return entry.get("value")
    return entry


def confidence_of(fields, name):
    entry = fields.get(name)
    if isinstance(entry, dict):
        try:
            return float(entry.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3, help="passes per document")
    ap.add_argument("--ground-truth", default=os.path.join(REPO_ROOT, "eval", "ground_truth.json"))
    ap.add_argument("--docs-dir", default=os.path.join(REPO_ROOT, "test-documents"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "ACCURACY.md"))
    ap.add_argument(
        "--model",
        default=None,
        help=(
            "Groq model for the eval run. Defaults to Config.GROQ_MODEL. Passing a "
            "different value overrides it for this run only, without editing "
            "config.py; the report then states that its numbers describe a "
            "different model than the code pins."
        ),
    )
    ap.add_argument(
        "--reasoning-model",
        default=None,
        help="Override Config.GROQ_MODEL_REASONING (router text, query answer).",
    )
    args = ap.parse_args()

    # Runtime-only override. The agents read Config.GROQ_MODEL at call time, so
    # setting it here reaches all three without touching any service file.
    import config

    configured = config.Config.GROQ_MODEL
    if args.model is None:
        args.model = configured
    if args.model != configured:
        print(
            f"NOTE: overriding Config.GROQ_MODEL {configured!r} -> {args.model!r} "
            f"for this run only (config.py is unmodified).",
            flush=True,
        )
    config.Config.GROQ_MODEL = args.model
    if args.reasoning_model:
        config.Config.GROQ_MODEL_REASONING = args.reasoning_model
    MODEL_NOTE["configured"] = configured
    MODEL_NOTE["used"] = args.model
    MODEL_NOTE["reasoning"] = config.Config.GROQ_MODEL_REASONING

    # The module already chdir'd into ai-service/, so resolve any relative path the
    # caller passed against the directory they actually invoked from.
    args.ground_truth = os.path.join(INVOKED_FROM, args.ground_truth)
    args.docs_dir = os.path.join(INVOKED_FROM, args.docs_dir)
    args.out = os.path.join(INVOKED_FROM, args.out)

    with open(args.ground_truth, encoding="utf-8") as fh:
        truth = json.load(fh)

    documents = truth["documents"]
    for doc in documents:
        if not doc.get("customer_id"):
            raise SystemExit(
                f"{doc['file_name']}: customer_id is empty. An empty value falls back "
                f"to 'generic' rules, which would make the decision measurement "
                f"meaningless. Fill it in before running."
            )
        canonical_decision(doc["expected_decision"])  # fail fast on bad labels

    results = defaultdict(list)
    paths = {}

    for doc in documents:
        name = doc["file_name"]
        file_path = os.path.join(args.docs_dir, name)
        if not os.path.exists(file_path):
            raise SystemExit(f"Missing document: {file_path}")
        paths[name] = detect_path(file_path)
        for run_index in range(args.runs):
            print(f"  [{run_index + 1}/{args.runs}] {name} ({paths[name]})", flush=True)
            results[name].append(run_document(file_path, doc["customer_id"]))

    report = build_report(documents, results, paths, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"\nWrote {args.out}")

    query_calls = measure_query_agent(args)
    cost_out = os.path.join(os.path.dirname(args.out), "COST.md")
    with open(cost_out, "w", encoding="utf-8") as fh:
        fh.write(build_cost_report(documents, results, paths, args, query_calls))
    print(f"Wrote {cost_out}")


# Cost probes for the Query Agent. It is not part of /process, so it never appears
# in a per-document cost. These questions exercise its two calls; they are cost
# probes, not correctness tests, and are not ground truth.
QUERY_PROBES = [
    "How many shipments are there?",
    "Which shipments are from Shanghai?",
    "What is the average gross weight?",
]


def measure_query_agent(args):
    """Run the query agent once per probe, recording its two calls each time."""
    from services.query_agent import run_query

    collected = []
    for question in QUERY_PROBES:
        llm_metrics.reset()
        try:
            run_query(question)
        except Exception as exc:  # a failed probe still has recorded calls
            print(f"  query probe failed ({question!r}): {exc}", flush=True)
        collected.extend(llm_metrics.get_calls())
    return collected


def _fmt_usd(value):
    return f"${value:.6f}"


def build_cost_report(documents, results, paths, args, query_calls):
    lines = []
    w = lines.append
    now = datetime.now(timezone.utc)

    doc_calls = [c for doc in documents for r in results[doc["file_name"]] for c in r["llm_calls"]]
    doc_summary = llm_metrics.summarise(doc_calls)
    runs = args.runs
    n_docs = len(documents)
    doc_runs = n_docs * runs

    w("# LLM cost baseline")
    w("")
    w(f"Generated {now.strftime('%Y-%m-%d %H:%M UTC')} by `eval/run_eval.py`.")
    w("")
    w("**This is the pre-optimisation baseline. No cost reduction has been applied.**")
    w("")
    w(f"Extraction and NL->SQL model: `{MODEL_NOTE['used']}`. "
      f"Router reasoning and query answer: `{MODEL_NOTE['reasoning']}`. "
      "Token counts are taken from each Groq response's `usage` field, never "
      "counted locally.")
    w("")
    w(f"Pricing source: {llm_metrics.PRICING_SOURCE}")
    w("")
    for m in dict.fromkeys([MODEL_NOTE["used"], MODEL_NOTE["reasoning"]]):
        pr = llm_metrics.PRICING_PER_1M.get(m)
        if pr:
            w(f"- `{m}`: ${pr['input']:.3f} per 1M input, ${pr['output']:.2f} per 1M output")
        else:
            w(f"- `{m}`: no published price — recorded as unpriced, never as zero")
    w("")

    w("## Cost per document processed")
    w("")
    total_cost = doc_summary["total"]["cost_usd"]
    w(f"**{_fmt_usd(total_cost / doc_runs)} per document**, averaged over "
      f"{doc_runs} runs ({n_docs} documents x {runs} runs).")
    w("")
    w(f"Total for all {doc_runs} runs: {_fmt_usd(total_cost)} across "
      f"{doc_summary['total']['calls']} LLM calls, "
      f"{doc_summary['total']['prompt_tokens']:,} prompt tokens and "
      f"{doc_summary['total']['completion_tokens']:,} completion tokens.")
    w("")
    if doc_summary["total"]["failures"]:
        w(f"{doc_summary['total']['failures']} call(s) failed and are recorded with "
          "zero tokens.")
        w("")

    w("### Per agent, per document run")
    w("")
    w("| Agent | Calls/doc | Prompt tokens | Completion tokens | Cost/doc | Share |")
    w("|---|---|---|---|---|---|")
    for agent in sorted(doc_summary["per_agent"], key=lambda a: -doc_summary["per_agent"][a]["cost_usd"]):
        b = doc_summary["per_agent"][agent]
        share = (b["cost_usd"] / total_cost * 100) if total_cost else 0
        w(f"| `{agent}` | {b['calls'] / doc_runs:.1f} | "
          f"{b['prompt_tokens'] / doc_runs:,.0f} | {b['completion_tokens'] / doc_runs:,.0f} | "
          f"{_fmt_usd(b['cost_usd'] / doc_runs)} | {share:.0f}% |")
    w("")
    w("Prompt and completion token columns are per document run. The validator "
      "makes no LLM call at all and so does not appear.")
    w("")

    w("### Per document")
    w("")
    w("| Document | Path | Prompt tokens | Completion tokens | Cost |")
    w("|---|---|---|---|---|")
    for doc in documents:
        name = doc["file_name"]
        calls = [c for r in results[name] for c in r["llm_calls"]]
        s = llm_metrics.summarise(calls)["total"]
        w(f"| `{name}` | {paths[name]} | {s['prompt_tokens'] / runs:,.0f} | "
          f"{s['completion_tokens'] / runs:,.0f} | {_fmt_usd(s['cost_usd'] / runs)} |")
    w("")
    w("Averaged per run. Cost tracks prompt size, so the OCR document is not "
      "necessarily the most expensive despite being by far the slowest.")
    w("")

    w("## Query agent")
    w("")
    w("The Query Agent is reached through `POST /query`, not `/process`, so it "
      "contributes nothing to the per-document cost above. Measured separately "
      f"over {len(QUERY_PROBES)} probe questions.")
    w("")
    if query_calls:
        q_summary = llm_metrics.summarise(query_calls)
        n_q = len(QUERY_PROBES)
        w("| Step | Calls/question | Prompt tokens | Completion tokens | Cost/question |")
        w("|---|---|---|---|---|")
        for agent in sorted(q_summary["per_agent"]):
            b = q_summary["per_agent"][agent]
            w(f"| `{agent}` | {b['calls'] / n_q:.1f} | {b['prompt_tokens'] / n_q:,.0f} | "
              f"{b['completion_tokens'] / n_q:,.0f} | {_fmt_usd(b['cost_usd'] / n_q)} |")
        w(f"| **total** | {q_summary['total']['calls'] / n_q:.1f} | "
          f"{q_summary['total']['prompt_tokens'] / n_q:,.0f} | "
          f"{q_summary['total']['completion_tokens'] / n_q:,.0f} | "
          f"**{_fmt_usd(q_summary['total']['cost_usd'] / n_q)}** |")
        w("")
        w("Probe questions: " + ", ".join(f"`{q}`" for q in QUERY_PROBES) + ". These "
          "measure cost, not answer quality — there is no ground truth for them.")
    else:
        w("No query-agent calls were recorded.")
    w("")

    w("## Method")
    w("")
    w("- Every Groq call goes through the instrumented proxy in "
      "`ai-service/utils/llm_metrics.py`, which records the response's `usage` "
      "fields, wall-clock latency and computed cost. It forwards to Groq "
      "unchanged: no prompt, model or behaviour is altered by instrumentation.")
    w("- Cost is `prompt_tokens/1e6 * input_price + completion_tokens/1e6 * "
      "output_price`, using the published prices cited above.")
    w("- A model with no published price records `None`, never `0`, so an "
      "unpriced call cannot silently look free.")
    w("")
    w(f"Reproduce with `python eval/run_eval.py --runs {runs}`.")
    w("")
    return "\n".join(lines)


def build_report(documents, results, paths, args):
    runs = args.runs
    scored_fields = list(REQUIRED_FIELDS)

    # ── Field accuracy, scored on run 1; later runs feed the stability check ──
    per_field = {f: {"correct": 0, "total": 0} for f in scored_fields}
    per_doc_rows = []
    calibration = {b: {"correct": 0, "total": 0} for b in CONFIDENCE_BUCKETS}
    unstable = []
    extra_labels = set()

    for doc in documents:
        name = doc["file_name"]
        runs_out = results[name]
        first = runs_out[0]
        expected = doc["expected_fields"]
        extra_labels |= set(expected) - set(scored_fields)

        doc_correct = 0
        for fname in scored_fields:
            if fname not in expected:
                continue
            actual = value_of(first["fields"], fname)
            ok, _ = fields_match(fname, expected[fname], actual)
            per_field[fname]["total"] += 1
            per_field[fname]["correct"] += int(ok)
            doc_correct += int(ok)

            bucket = bucket_of(confidence_of(first["fields"], fname))
            if bucket:
                calibration[bucket]["total"] += 1
                calibration[bucket]["correct"] += int(ok)

            values = {
                normalise_text(value_of(r["fields"], fname)) for r in runs_out
            }
            if len(values) > 1:
                unstable.append(
                    (name, fname, [value_of(r["fields"], fname) for r in runs_out])
                )

        per_doc_rows.append(
            {
                "name": name,
                "path": paths[name],
                "correct": doc_correct,
                "total": sum(1 for f in scored_fields if f in expected),
                "expected_decision": canonical_decision(doc["expected_decision"]),
                "decisions": [r["decision"] for r in runs_out],
                "seconds": [r["total_seconds"] for r in runs_out],
                "extraction_seconds": [r["extraction_seconds"] for r in runs_out],
                "retries": [r["retry_count"] for r in runs_out],
            }
        )

    # ── Decision confusion matrix (run 1) ──
    matrix = Counter()
    for row in per_doc_rows:
        matrix[(row["expected_decision"], row["decisions"][0])] += 1
    decision_correct = sum(v for (e, a), v in matrix.items() if e == a)

    lines = []
    w = lines.append
    now = datetime.now(timezone.utc)
    total_fields = sum(v["total"] for v in per_field.values())
    total_correct = sum(v["correct"] for v in per_field.values())

    w("# Extraction accuracy")
    w("")
    w(f"Generated {now.strftime('%Y-%m-%d %H:%M UTC')} by `eval/run_eval.py`.")
    w("")
    w("## Read this first")
    w("")
    w(
        f"**This set is {len(documents)} documents, hand-labelled by one person.** "
        "The numbers below are indicative, not measurements of general performance. "
        "Counts are reported with their denominators rather than as percentages, "
        "because a percentage over three documents implies a precision that does "
        "not exist here."
    )
    w("")
    w(
        "The confidence-calibration buckets are especially thin — several are "
        "likely empty or hold a single field. Read them as a sanity check on "
        "whether self-reported confidence means anything at all, not as a "
        "calibration curve."
    )
    w("")
    w(f"Each document was run **{runs} times** (the extractor uses temperature 0.1, "
      "so output can vary). Accuracy is scored on run 1; the extra runs feed the "
      "stability check at the end.")
    w("")
    w(f"Model used: **`{MODEL_NOTE['used']}`**.")
    w("")
    if MODEL_NOTE["configured"] != MODEL_NOTE["used"]:
        w(f"> **The model is not the one the code pins.** `config.py:42` hardcodes "
          f"`GROQ_MODEL = \"{MODEL_NOTE['configured']}\"`, which returns "
          f"`404 model_not_found` — Groq has decommissioned it, and no Llama chat "
          f"model is available on this account. The pipeline therefore cannot run "
          f"as configured: every extraction attempt fails and the extractor returns "
          f"null fields. This run overrides the model at runtime "
          f"(`--model {MODEL_NOTE['used']}`) purely so a baseline exists; "
          f"`config.py` is unmodified. **Every number below describes "
          f"`{MODEL_NOTE['used']}`, not `{MODEL_NOTE['configured']}`.**")
        w("")

    w("## Field extraction accuracy")
    w("")
    w(f"**{total_correct} of {total_fields} field values correct** across "
      f"{len(documents)} documents.")
    w("")
    w("| Field | Correct | Comparison | Why |")
    w("|---|---|---|---|")
    reasons = {
        "hs_code": "structured code; punctuation normalised exactly as the validator does",
        "incoterms": "closed three-letter vocabulary; a near-miss is a different term",
        "invoice_number": "identifier; one character off is a different invoice",
        "consignee_name": "free text; case/spacing/punctuation vary by document",
        "port_of_loading": "free text; e.g. trailing punctuation varies",
        "port_of_discharge": "free text; e.g. trailing punctuation varies",
        "description_of_goods": "free text; wording and casing vary",
        "gross_weight": "free text; unit spacing and punctuation vary",
    }
    for fname in scored_fields:
        stats = per_field[fname]
        kind = "exact" if fname in EXACT_FIELDS else "normalised"
        if fname == "hs_code":
            kind = "exact (HS normalised)"
        w(f"| `{fname}` | {stats['correct']} of {stats['total']} | {kind} | "
          f"{reasons.get(fname, '')} |")
    w("")
    w("Normalised comparison ignores case, collapses whitespace and strips "
      "punctuation. Exact comparison ignores only surrounding whitespace and case. "
      "`hs_code` uses the validator's own normalisation "
      "(`.replace('.', '').replace(' ', '')`, `services/validator_agent.py:251`) so "
      "scoring cannot drift from validation.")
    w("")
    if extra_labels:
        w(f"Excluded from scoring: {', '.join('`%s`' % f for f in sorted(extra_labels))} "
          "— labelled in the ground truth but not in `REQUIRED_FIELDS` "
          "(`services/ai_service.py:29-38`), so the extractor has no concept of it "
          "and it could never match.")
        w("")

    w("### Per document")
    w("")
    w("| Document | Path | Fields correct |")
    w("|---|---|---|")
    for row in per_doc_rows:
        w(f"| `{row['name']}` | {row['path']} | {row['correct']} of {row['total']} |")
    w("")

    w("## Decision accuracy")
    w("")
    w(f"**{decision_correct} of {len(documents)} decisions correct** (run 1).")
    w("")
    w("Rows are the hand-labelled expectation, columns the router's output.")
    w("")
    w("| expected \\ actual | " + " | ".join(f"`{d}`" for d in DECISIONS) + " |")
    w("|---|" + "---|" * len(DECISIONS))
    for exp in DECISIONS:
        cells = [str(matrix.get((exp, act), 0)) for act in DECISIONS]
        w(f"| `{exp}` | " + " | ".join(cells) + " |")
    w("")
    w("Ground-truth labels were written in prose and mapped onto the constants in "
      "`services/router_agent.py`: "
      + ", ".join(f"`{k}` → `{v}`" for k, v in sorted(DECISION_ALIASES.items()))
      + ". An unrecognised label aborts the run rather than scoring as a miss.")
    w("")

    w("## Confidence calibration")
    w("")
    w("Every extracted field bucketed by the confidence the extractor reported for "
      "it, against whether that field was actually correct. This tests whether "
      "self-reported confidence predicts correctness at all — which every "
      "`confidence_threshold` in `customer_rules.json` assumes.")
    w("")
    w("| Confidence bucket | Fields correct |")
    w("|---|---|")
    for low, high in CONFIDENCE_BUCKETS:
        stats = calibration[(low, high)]
        cell = (
            f"{stats['correct']} of {stats['total']}" if stats["total"] else "no fields"
        )
        w(f"| {low:.2f} – {high:.2f} | {cell} |")
    w("")
    populated = [b for b in CONFIDENCE_BUCKETS if calibration[b]["total"]]
    w(f"{len(populated)} of {len(CONFIDENCE_BUCKETS)} buckets contain any fields. "
      "With this few documents the buckets cannot show a calibration trend; they "
      "show only whether confidence and correctness are wildly inconsistent.")
    w("")

    w("## Latency")
    w("")
    w("Wall-clock per run. `extract` is text extraction only (the PyMuPDF or "
      "PaddleOCR step); `total` includes the extractor LLM call, validation and "
      "the router LLM call.")
    w("")
    w("| Document | Path | extract (s) | total (s) |")
    w("|---|---|---|---|")
    for row in per_doc_rows:
        ex = ", ".join(f"{s:.2f}" for s in row["extraction_seconds"])
        tot = ", ".join(f"{s:.2f}" for s in row["seconds"])
        w(f"| `{row['name']}` | {row['path']} | {ex} | {tot} |")
    w("")
    by_path = defaultdict(list)
    for row in per_doc_rows:
        by_path[row["path"]].extend(row["extraction_seconds"])
    for path_name in ("digital", "ocr"):
        vals = by_path.get(path_name)
        if vals:
            docs_n = sum(1 for r in per_doc_rows if r["path"] == path_name)
            w(f"- **{path_name}**: {len(vals)} runs over {docs_n} document(s), "
              f"text extraction {min(vals):.2f}–{max(vals):.2f}s")
    w("")
    counts = Counter(r["path"] for r in per_doc_rows)
    if counts.get("ocr", 0) <= 1 or counts.get("digital", 0) <= 1:
        w("The two paths are backed by very few documents "
          f"({counts.get('digital', 0)} digital, {counts.get('ocr', 0)} OCR), so this "
          "split is a single observation per path rather than a comparison.")
        w("")

    w("## Run-to-run stability")
    w("")
    if unstable:
        w(f"**{len(unstable)} field(s) changed between runs.** The extractor runs at "
          "temperature 0.1, not 0, so identical input can produce different output. "
          "This is a finding in itself: the same document can validate differently "
          "on different days.")
        w("")
        w("| Document | Field | Values across runs |")
        w("|---|---|---|")
        for name, fname, values in unstable:
            rendered = " / ".join(repr(v) for v in values)
            w(f"| `{name}` | `{fname}` | {rendered} |")
    else:
        w(f"No field changed across {runs} runs of each document. That is not proof "
          "of determinism — temperature is 0.1, not 0 — only that no variation "
          "surfaced in this many runs.")
    w("")

    w("## Appendix — every field, expected vs extracted (run 1)")
    w("")
    w("With a set this small the individual misses say more than the totals.")
    w("")
    for doc in documents:
        name = doc["file_name"]
        first = results[name][0]
        w(f"**`{name}`** ({paths[name]} path)")
        w("")
        w("| Field | Expected | Extracted | Confidence | |")
        w("|---|---|---|---|---|")
        for fname in scored_fields:
            if fname not in doc["expected_fields"]:
                continue
            exp = doc["expected_fields"][fname]
            act = value_of(first["fields"], fname)
            conf = confidence_of(first["fields"], fname)
            ok, _ = fields_match(fname, exp, act)
            w(f"| `{fname}` | {exp or '—'} | {act if act not in (None, '') else '—'} "
              f"| {conf:.2f} | {'ok' if ok else '**miss**'} |")
        w("")

    w("## Method")
    w("")
    w("- Documents are read from `test-documents/` on local disk; S3 is not used.")
    w("- Each document runs the real pipeline: `smart_extraction_pipeline` → "
      "`extract_shipment_fields` → `validate_shipment` → `route_decision`.")
    w("- Ground truth is `eval/ground_truth.json`, hand-labelled. This harness "
      "never writes to it.")
    w("- Extraction path is determined by re-running the same check "
      "`smart_extraction_pipeline` uses, so attribution matches the pipeline's own "
      "routing.")
    w("")
    w("Reproduce with `python eval/run_eval.py --runs %d`." % runs)
    w("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
