# Multi-Agent Trade Document Processing Platform

An AI-powered platform that automates trade document processing using a **4-agent pipeline architecture**. Upload a Commercial Invoice, Bill of Lading, or Packing List — the system extracts shipment fields via OCR + LLM, validates them against customer-specific rules, routes a compliance decision, and lets you query the data in natural language.

The pipeline is measured, not asserted: a 150-test suite covers the deterministic logic, an evaluation harness scores extraction against hand-labelled ground truth, and every LLM call is instrumented for tokens, latency and cost. Measured results live in [`docs/`](docs/) — including a caching experiment that was measured and **rejected**.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Agent Pipeline](#agent-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Setup & Run Instructions](#setup--run-instructions)
- [How It Works — End to End](#how-it-works--end-to-end)
- [Customer Rules Engine](#customer-rules-engine)
- [OCR & Text Extraction Strategy](#ocr--text-extraction-strategy)
- [Query Agent — Natural Language to SQL](#query-agent--natural-language-to-sql)
- [API Endpoints](#api-endpoints)
- [Key Design Decisions](#key-design-decisions)
- [Testing](#testing)
- [Evaluation & Accuracy](#evaluation--accuracy)
- [LLM Cost Engineering](#llm-cost-engineering)
- [Caching](#caching)
- [Reproducing Every Number](#reproducing-every-number)
- [Known Findings](#known-findings)
- [Tested Sample Documents](#example-test-cases)

---

## Architecture Overview

The platform follows a **three-tier architecture** with a clear separation of concerns:

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────────────┐
│                 │     │                  │     │         AI Service (Python)      │
│  React Frontend │────▶│  Node.js Server  │────▶│                                 │
│  (Vite + React) │     │  (Express API)   │     │  ┌───────────┐  ┌───────────┐   │
│                 │◀────│                  │◀────│  │ Extractor │─▶│ Validator │   │
│  Dashboard      │     │  - S3 upload     │     │  │   Agent   │  │   Agent   │   │
│  Shipment List  │     │  - Auth layer    │     │  └───────────┘  └─────┬─────┘   │
│  Query Panel    │     │  - Proxy to AI   │     │                       │         │
│  Detail View    │     │                  │     │                 ┌─────▼─────┐   │
│                 │     │                  │     │                 │  Router   │   │
└─────────────────┘     └──────────────────┘     │                 │   Agent   │   │
                                                 │                 └───────────┘   │
                                                 │  ┌───────────┐                 │
                                                 │  │   Query   │  ┌───────────┐  │
                                                 │  │   Agent   │──│  SQLite   │  │
                                                 │  └───────────┘  └───────────┘  │
                                                 └─────────────────────────────────┘
```

| Layer | Technology | Role |
|-------|-----------|------|
| **Frontend** | React 19 + Vite + TailwindCSS | Upload UI, dashboard, shipment detail panel, NL query interface |
| **Backend** | Node.js + Express 5 | File upload to S3, auth middleware, API proxy to Python AI service |
| **AI Service** | Python + FastAPI + Groq `gpt-oss-120b` / `gpt-oss-20b` | 4-agent pipeline: OCR → Extract → Validate → Route → Query |
| **Storage** | SQLite (shipment data) + AWS S3 (raw files) | Zero-infrastructure persistence — SQLite file travels with the service |

---

## Agent Pipeline

The core of the platform is a **sequential multi-agent pipeline** where each agent has a single responsibility:

### 1. Extractor Agent (`services/ai_service.py`)

- **Input**: Raw text from OCR/PDF extraction
- **Output**: 8 structured shipment fields, each with `{value, confidence, source_evidence}`
- **LLM**: Groq `openai/gpt-oss-120b` with `temperature=0.1` for near-deterministic extraction
- **Fields extracted**: `consignee_name`, `hs_code`, `port_of_loading`, `port_of_discharge`, `incoterms`, `description_of_goods`, `gross_weight`, `invoice_number`
- **Anti-hallucination**: LLM is instructed to return `null` for uncertain fields — never guess
- **Retry logic**: Up to 2 retries on LLM failure with full audit logging

### 2. Validator Agent (`services/validator_agent.py`)

- **Input**: Extracted fields from Agent 1 + customer ID
- **Output**: Per-field validation status — `match` | `mismatch` | `uncertain`
- **Zero LLM calls** — entirely deterministic Python logic for auditability
- **Customer rules** loaded from `customer_rules.json` (configurable per customer)
- **Fuzzy matching** via `difflib.SequenceMatcher` to handle OCR misreads (e.g., "Aphle" → "Apple" detected as similar, routed to review instead of hard rejection)
- **Confidence gating**: Fields below the customer-specific threshold are auto-flagged as `uncertain`

### 3. Router Agent (`services/router_agent.py`)

- **Input**: Validation output from Agent 2
- **Output**: Final decision + human-readable reasoning
- **Hybrid design**: Decision is **deterministic** (rule-based), reasoning is **LLM-generated**
- **Decision rules**:
  - Any mismatch → `amendment_required`
  - Any uncertain field (no mismatches) → `human_review`
  - All matched → `auto_approve`
- **LLM role**: Generates professional compliance explanation, amendment drafts, and approval summaries — the decision itself never depends on the LLM

### 4. Query Agent (`services/query_agent.py`)

- **Input**: Natural language question from the user
- **Output**: SQL-grounded answer with raw data
- **Pipeline**: Question → LLM → SQL → SQLite execution → LLM → natural language answer
- **Anti-hallucination**: The answer LLM receives **only the actual query results**, never guesses
- **Security**: Only `SELECT` statements are permitted — no writes allowed

---

## Tech Stack

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 19.x | UI framework |
| Vite | 5.x | Build tool and dev server |
| TailwindCSS | 3.x | Utility-first styling |
| Axios | 1.x | HTTP client |

### Backend (Node.js)
| Technology | Version | Purpose |
|-----------|---------|---------|
| Express | 5.x | HTTP server and routing |
| AWS SDK v3 | 3.x | S3 file upload and presigned URL generation |
| Multer | 2.x | Multipart file upload handling |
| Axios | 1.x | Proxy requests to Python AI service |

### AI Service (Python)
| Technology | Purpose |
|-----------|---------|
| FastAPI + Uvicorn | Async HTTP server |
| Groq SDK | LLM inference (`openai/gpt-oss-120b` extraction, `openai/gpt-oss-20b` reasoning) |
| PaddleOCR (PP-OCRv3) | OCR for scanned/handwritten documents |
| PyMuPDF (fitz) | Digital PDF text extraction |
| OpenCV | Image preprocessing for OCR |
| SQLite3 | Embedded database for shipment data |
| Boto3 | AWS S3 file downloads |
| pytest | Test suite (dev only — not in the production image) |
| fastembed | Local embeddings for the semantic-cache experiment (dev only) |

---

## Project Structure

```
GoComet/
├── client/                        # React Frontend
│   ├── src/
│   │   ├── App.jsx                # Main application — dashboard, upload, views
│   │   ├── components/
│   │   │   ├── FieldCard.jsx      # Extracted field display with confidence bar
│   │   │   ├── PipelineStatus.jsx # Visual pipeline stepper (5 stages)
│   │   │   ├── QueryPanel.jsx     # Natural language query interface
│   │   │   ├── RouterDecision.jsx # Decision card with reasoning + amendments
│   │   │   ├── ShipmentList.jsx   # Shipment table with status badges
│   │   │   └── ValidationTable.jsx# Field-by-field validation results
│   │   ├── index.css              # Global styles
│   │   └── main.jsx               # React entry point
│   ├── .env                       # VITE_API_URL and Firebase config
│   └── package.json
│
├── server/                        # Node.js Backend
│   ├── server.js                  # Express API — upload, trigger, proxy
│   ├── .env                       # AWS + Groq + Python service URL
│   └── package.json
│
├── ai-service/                    # Python AI Service
│   ├── main.py                    # FastAPI app — endpoints + pipeline orchestration
│   ├── config.py                  # Environment config + validation
│   ├── logger.py                  # Structured logging
│   ├── customer_rules.json        # Per-customer validation rules (6 customers)
│   ├── requirements.txt           # Python dependencies
│   ├── services/
│   │   ├── ai_service.py          # Extractor Agent — LLM field extraction
│   │   ├── validator_agent.py     # Validator Agent — deterministic rule checks
│   │   ├── router_agent.py        # Router Agent — decision + LLM reasoning
│   │   ├── query_agent.py         # Query Agent — NL → SQL → answer
│   │   ├── extraction_service.py  # Smart extraction: digital PDF vs OCR routing
│   │   └── ocr_service.py         # PaddleOCR pipeline (PP-OCRv3, CPU-optimized)
│   ├── utils/
│   │   ├── db_utils.py            # SQLite CRUD — shipments, validations, decisions, audit
│   │   ├── s3_utils.py            # AWS S3 download utility
│   │   ├── llm_metrics.py         # Per-call token/latency/cost recording + cost breaker
│   │   ├── ocr_cache.py           # OCR text cache, keyed on document bytes
│   │   └── sql_cache.py           # NL→SQL translation cache, keyed on question + schema
│   ├── tests/                     # pytest suite — 150 tests, no network, no API keys
│   │   ├── conftest.py            # Fixtures: dummy key, throwaway DB, rules path
│   │   ├── test_validator_agent.py   # Rule types, confidence boundary, fuzzy cutoff
│   │   ├── test_router_agent.py      # Decision precedence, confidence floors
│   │   ├── test_sql_guard.py         # SELECT-only guard, injection, stacked statements
│   │   ├── test_extraction_routing.py# Digital-PDF vs OCR threshold boundaries
│   │   ├── test_llm_metrics.py       # Cost maths, attribution, circuit breaker
│   │   ├── test_caches.py            # OCR + SQL cache behaviour and invalidation
│   │   └── test_eval_comparators.py  # Eval scoring agrees with validator logic
│   ├── requirements.txt           # Production dependencies
│   ├── requirements-dev.txt       # pytest + fastembed (never in the Docker image)
│   └── shipments.db               # SQLite database (auto-created on startup, gitignored)
│
├── eval/                          # Evaluation harness + hand-labelled ground truth
│   ├── run_eval.py                # Scores extraction, decisions, calibration, latency, cost
│   ├── sweep_semantic.py          # Cosine threshold sweep for the semantic cache tier
│   ├── ground_truth.json          # Hand-labelled expected values (never machine-written)
│   ├── ground_truth.template.json # Empty template, kept pristine
│   ├── query_pairs.json           # Labelled question pairs for the sweep
│   ├── query_pairs.template.json  # Empty template, kept pristine
│   ├── README.md                  # How to label ground truth, and why by hand
│   └── README-query-pairs.md      # How to label query pairs
│
├── docs/                          # Measured results (regenerated, not hand-written)
│   ├── ACCURACY.md                # Field/decision accuracy, calibration, latency
│   ├── COST.md                    # Per-document LLM cost, per agent
│   ├── SEMANTIC_CACHE.md          # Threshold sweep — the negative result
│   └── FINDINGS.md                # 8 defects found while adding coverage
│
├── test-documents/                # The 3-document evaluation set
│
└── README.md
```

---

## Prerequisites

Ensure you have the following installed:

| Requirement | Minimum Version | Check Command |
|------------|----------------|--------------|
| **Node.js** | 18.x or higher | `node -v` |
| **npm** | 9.x or higher | `npm -v` |
| **Python** | 3.10 or higher | `python3 --version` |
| **pip** | Latest | `pip --version` |

You will also need:
- An **AWS account** with an S3 bucket for file storage
- A **Groq API key** (free tier available at [console.groq.com](https://console.groq.com))

---

## Environment Variables

### `server/.env`

```env
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
AWS_BUCKET_NAME=your_s3_bucket_name
GROQ_API_KEY=your_groq_api_key
PYTHON_SERVICE_URL=http://127.0.0.1:7860
```

### `ai-service/.env`

```env
GROQ_API_KEY=your_groq_api_key
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=your_aws_region
AWS_BUCKET_NAME=your_s3_bucket_name
```

### `client/.env`

```env
VITE_API_URL=http://localhost:5001
```

---

## Setup & Run Instructions

> **All three services must be running simultaneously.** Open three separate terminal windows/tabs.

### Terminal 1 — Python AI Service (Port 7860)

```bash
cd ai-service

# Create virtual environment (first time only)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# Install dependencies (first time only)
pip install -r requirements.txt

# Start the AI service
python main.py
```

> **Note:** On first run, PaddleOCR will download ~150MB of model weights. This is a one-time operation. The service will print `[TradeAI] SQLite database initialized` when ready.

### Terminal 2 — Node.js Backend (Port 5001)

```bash
cd server

# Install dependencies (first time only)
npm install

# Start the server
node server.js
```

> You should see: `[Server] TradeAI Backend running on port 5001`

### Terminal 3 — React Frontend (Port 5173)

```bash
cd client

# Install dependencies (first time only)
npm install

# Start dev server
npm run dev
```

> Open **http://localhost:5173** in your browser.

### Verify Everything Is Working

Once all three terminals are running:
1. Open `http://localhost:5173` — you should see the GoComet dashboard
2. The sidebar should show **System Online** with a green indicator
3. Quick Stats should display zeros (no shipments yet)
4. Upload a trade document PDF and select a customer → the pipeline will process it in ~15–30 seconds

---

## How It Works — End to End

Here's the complete flow when you upload a document:

```
User uploads PDF via React UI
        │
        ▼
Node.js receives file via Multer
        │
        ▼
File uploaded to AWS S3 (raw storage)
        │
        ▼
Node.js calls POST /trigger → Python AI service
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                PYTHON AI PIPELINE                     │
│                                                       │
│  1. Download file from S3                             │
│  2. Smart Text Extraction                             │
│     ├── Digital PDF? → PyMuPDF (fast path)            │
│     └── Scanned/Image? → PaddleOCR (PP-OCRv3)        │
│  3. Extractor Agent → 8 fields via Groq LLM          │
│  4. Validator Agent → rule-based field validation     │
│  5. Router Agent → decision + LLM reasoning           │
│  6. All results saved to SQLite                       │
│  7. Every step logged to audit_logs                   │
└───────────────────────────────────────────────────────┘
        │
        ▼
React UI polls /shipments every 5s → shows results
```

---

## Customer Rules Engine

The Validator Agent loads rules from `ai-service/customer_rules.json`. Each customer has:

| Rule | Example (Apple) | Purpose |
|------|----------------|---------|
| `required_incoterms` | `["DAP", "DDP"]` | Allowed trade terms |
| `allowed_ports_of_loading` | `["Shenzhen", "Hong Kong", ...]` | Valid origin ports |
| `allowed_consignees` | `["Apple Inc.", "Apple Operations International"]` | Expected consignee names |
| `required_hs_code_prefix` | `"8471"` | HS code must start with this prefix |
| `confidence_threshold` | `0.72` | Minimum extraction confidence to accept a field |

**Pre-configured customers:** Nike, Adidas, Zara, Apple, Maersk, and a Generic fallback.

To add a new customer, add a new entry to the `customers` object in `customer_rules.json`. The validator will pick it up on the next pipeline run without a restart.

---

## OCR & Text Extraction Strategy

The system uses a **hybrid extraction strategy** that minimizes OCR usage for speed:

```
Input file
    │
    ├── Is it an image (PNG/JPG)?
    │       └── Yes → PaddleOCR directly
    │
    └── Is it a PDF?
            │
            ├── Try PyMuPDF digital text extraction (fast path)
            │       └── Text quality check: ≥80 chars + ≥25% alphabetic?
            │               ├── Yes → Use digital text (instant)
            │               └── No → Fallback to PaddleOCR
            │
            └── PaddleOCR Pipeline:
                    1. Convert PDF page → image at 130 DPI
                    2. Resize to 1000px width (color-preserving)
                    3. Run PP-OCRv3 (CPU-optimized)
                    4. If <100 chars extracted → retry at 1400px
```

**Why PaddleOCR over Tesseract?** PP-OCRv3 achieves significantly higher accuracy on mixed-language and handwritten documents, which is common in trade documentation.

---

## Query Agent — Natural Language to SQL

The Query panel lets you ask questions in plain English. Examples:

| Question | What happens |
|----------|-------------|
| "How many shipments were processed today?" | → `SELECT COUNT(*) FROM shipments WHERE date(created_at) = date('now')` |
| "Show all Apple shipments that need amendment" | → `SELECT * FROM shipments s JOIN agent_decisions d ON ... WHERE d.decision = 'amendment_required'` |
| "What was the average extraction confidence?" | → `SELECT AVG(extraction_confidence) FROM shipments` |
| "List all mismatched fields" | → `SELECT * FROM validation_results WHERE status = 'mismatch'` |

The answer is always **grounded in actual database results** — the LLM cannot hallucinate numbers because it only sees real query output.

---

## API Endpoints

### Node.js Server (Port 5001)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Upload trade document to S3 |
| `POST` | `/trigger` | Trigger AI pipeline for a document |
| `GET` | `/documents` | List all uploaded documents |
| `GET` | `/documents/:id/view` | Get presigned S3 URL for document viewing |
| `GET` | `/shipments` | List all processed shipments (proxied to Python) |
| `GET` | `/shipments/:id` | Get full pipeline result for one shipment |
| `GET` | `/stats` | Dashboard statistics |
| `GET` | `/decisions` | List all router decisions |
| `POST` | `/query` | Natural language query over shipment data |

### Python AI Service (Port 7860)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/process` | Trigger 3-agent pipeline (async background task) |
| `GET` | `/shipments` | List shipments from SQLite |
| `GET` | `/shipments/:id` | Full shipment detail with validation + decision + audit trail |
| `GET` | `/stats` | Aggregate statistics |
| `GET` | `/decisions` | All agent decisions |
| `POST` | `/query` | NL → SQL → answer pipeline |
| `GET` | `/health` | Service health check |

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **SQLite over Firestore** | Zero infrastructure — the `.db` file is portable, embeddable, and supports raw SQL for the Query Agent. No cloud DB setup needed to run on a laptop. |
| **Deterministic validation (no LLM)** | Validator Agent uses pure Python logic — every result is reproducible and auditable. LLMs are non-deterministic and shouldn't make compliance decisions. |
| **Hybrid decision engine** | The Router's decision is deterministic (rule-based), but the explanation is LLM-generated. This gives you auditability AND readability. |
| **Fuzzy matching for OCR tolerance** | OCR can misread characters (e.g., "Aphle" vs "Apple"). Fuzzy matching with ≥70% similarity routes these to human review instead of outright rejection. |
| **Field-level confidence tracking** | Every extracted field carries a `confidence` score (0.0–1.0) and `source_evidence` (verbatim quote). This enables per-field audit and threshold-based gating. |
| **Full audit trail** | Every pipeline step — download, extraction, validation, routing — is logged to `audit_logs` with timestamps and durations. No silent failures. |
| **Smart OCR routing** | Digital PDFs skip OCR entirely (PyMuPDF fast path). Only scanned/image documents hit PaddleOCR. Measured: 0.01s on the digital path against 10.5–32s for OCR. |
| **Anti-hallucination in Query Agent** | The NL answer LLM receives only actual SQL results — it cannot invent data. Empty results explicitly return "no data found." |
| **Measure before optimising** | Every Groq call is instrumented for tokens/latency/cost *before* any reduction was attempted, so the 15% saving is a measured delta against a recorded baseline rather than an estimate. Token counts come from the API's `usage` field, never counted locally. |
| **Model routing by risk, not by cost alone** | The large model is kept where correctness matters (extraction, NL→SQL). Only steps that cannot affect a decision — the router's explanation text, the query answer phrasing — moved to the cheaper model. |
| **Cache the SQL, never the rows** | A result cache goes stale the moment a document is processed; a translation cache never does. Cache hits re-execute against live data. |
| **Cache keys bound to schema** | The SQL cache key includes a hash of the live `CREATE TABLE` DDL, so any schema change orphans every entry instead of serving SQL that references a dropped column. |
| **Cost cap fails loudly** | `CostLimitExceeded` is deliberately excluded from the retry and fallback paths that would otherwise absorb it. A breaker that gets retried and then silently returns nulls is not a breaker. |
| **Negative results are shipped as documentation** | The semantic cache tier was built, measured, and rejected because no threshold separated same-meaning from opposite-meaning questions. The experiment is recorded in `docs/SEMANTIC_CACHE.md` so it is not proposed again without the evidence. |
| **Findings are recorded, not silently fixed** | Defects found while adding coverage are documented with a measurement and pinned by a test asserting current behaviour, so a future change fails loudly rather than passing quietly. |

---

## Testing

**150 tests, covering the deterministic parts of the pipeline.** Agent decision logic is
never mocked into passing — where a test documents behaviour that is wrong, it is marked
as a tripwire and cross-referenced to [`docs/FINDINGS.md`](docs/FINDINGS.md).

```bash
cd ai-service
pip install -r requirements-dev.txt
pytest tests/ -v
```

| Test file | Covers |
|---|---|
| `test_validator_agent.py` | Each rule type against the real `customer_rules.json`; the confidence-threshold boundary at 0.74 / 0.75 / 0.76; the fuzzy cutoff bracketed with measured ratios; unknown customer → generic fallback |
| `test_router_agent.py` | Decision precedence (mismatch > uncertain > all-match) and the confidence floor clamps, tested either side to prove the clamp is live |
| `test_sql_guard.py` | `INSERT`/`UPDATE`/`DELETE`/`DROP` rejection, case and whitespace variants, stacked statements, and read-only queries the guard wrongly blocks |
| `test_extraction_routing.py` | The ≥80-character and ≥25%-alphabetic thresholds at three positions each; empty and whitespace input; path selection |
| `test_llm_metrics.py` | Cost arithmetic against published prices, per-agent attribution, and the cost circuit breaker |
| `test_caches.py` | Cache hit/miss, content-addressed keying, schema-change invalidation, corrupt-file survival |
| `test_eval_comparators.py` | That eval scoring cannot drift from the validator's own normalisation |

**No test touches the network.** The suite is verified to pass with all credentials
scrubbed from the environment and with sockets disabled — PaddleOCR, PyMuPDF and the Groq
client are all stubbed at the boundary.

---

## Evaluation & Accuracy

Full report: **[`docs/ACCURACY.md`](docs/ACCURACY.md)** — regenerated by the harness, not written by hand.

```bash
python eval/run_eval.py --runs 3
```

> **Read the denominators.** The evaluation set is **3 documents / 24 field values**,
> hand-labelled by one person. These numbers are indicative of behaviour, not a measure of
> general performance. The harness deliberately reports counts rather than percentages,
> because a rate over three documents implies a precision that does not exist.

| Metric | Result |
|---|---|
| Field extraction | **18 of 24** field values correct |
| — digital PDFs | 8 of 8 and 8 of 8 |
| — scanned document | **2 of 8** |
| Decisions | **2 of 3** correct |
| Text extraction, digital path | ~0.01s |
| Text extraction, OCR path | 10–60s, load-dependent |

**Every extraction failure is on the OCR path**, and all are character confusions:
`Apple`→`Aphle`, `847130`→`847I3O`, `APP-2026-001`→`APP-2O26-OO1`.

Ground truth is **hand-labelled only**. Populating it from pipeline output would score 100%
by construction and measure nothing — see [`eval/README.md`](eval/README.md). The harness
never writes to it.

The report also includes a confidence-calibration table, which is what surfaced
[finding 2](docs/FINDINGS.md): the extractor reported **0.95 confidence on OCR-corrupted
text**, so confidence predicts correctness on the digital path and fails on the OCR path —
where the `confidence_threshold` gating is most needed.

---

## LLM Cost Engineering

Full report: **[`docs/COST.md`](docs/COST.md)**

Every Groq call is wrapped by [`utils/llm_metrics.py`](ai-service/utils/llm_metrics.py),
which records **prompt tokens, completion tokens, model and latency from the API response's
`usage` field** — never counted locally — and computes cost from Groq's published
per-model pricing, cited with its source and fetch date in the module. A model with no
published price records as *unpriced*, never as zero.

### Measured reduction

| | Baseline | After | Change |
|---|---|---|---|
| Cost per document | $0.000988 | **$0.000838** | **−15%** |
| Cost per query | $0.000303 | $0.000276 | −9% |
| Field accuracy | 18 of 24 | 18 of 24 | **unchanged** |
| Decision accuracy | 2 of 3 | 2 of 3 | **unchanged** |

**Model routing** is the only change made. Extraction and NL→SQL stay on
`openai/gpt-oss-120b`; the router's reasoning and the query agent's answer step move to
`openai/gpt-oss-20b` at half price. Both are safe by construction — the router's decision is
already deterministic, and the answer step only phrases rows that have already been fetched.

### Prompt trimming: measured, then deliberately not done

| Prompt | Fixed boilerplate |
|---|---|
| Extractor | 88.9% |
| Router | 87.8% |
| Query NL→SQL | 99.1% |

Boilerplate dominates every prompt — but **input is only 21% of per-document cost**, because
output is priced 4× and the token volumes are similar. Deleting every boilerplate character
would save roughly 4% of total spend, so the prompts were left alone rather than risk
accuracy on prompts that currently extract correctly.

### Cost circuit breaker

`MAX_COST_PER_DOCUMENT_USD` (default **$0.05**, ~60× the measured baseline) is enforced by a
thread-local budget wrapping the pipeline. It is a runaway breaker, not a tuning knob.
Breaching it raises `CostLimitExceeded` and **fails the document loudly** — a cost cap is not
a transient LLM error, so it is deliberately excluded from the retry and fallback paths that
would otherwise absorb it. Cost is logged alongside latency on both success and failure.

---

## Caching

Two tiers ship. A third was measured and rejected.

Full benchmark: **[`docs/CACHE.md`](docs/CACHE.md)** — regenerated by `python eval/bench_cache.py`.

### 1. OCR text cache — keyed on document bytes

OCR is the slowest step by three orders of magnitude, and re-processing the same file
repeats it for a result that cannot have changed.

A cache hit replaces the entire OCR pass with a local file read. Measured across runs, it
removes **98–99%** of repeat OCR time.

> **On absolute timings.** Wall-clock OCR figures on this machine range from ~18s to ~60s
> for the *same* document depending on system load, so the benchmark reports medians with
> their full range and records the load average it ran under. The most recent run was taken
> at load average 222 and the report says on its face that its absolute numbers should not
> be cited — the **ratio** is the robust quantity, since cold and warm are inflated by the
> same contention. Re-run `python eval/bench_cache.py` on an idle machine for citable
> absolutes.

Keyed on SHA-256 of the file's bytes, so an edited document misses. Cache failures fall
through to real OCR rather than breaking extraction, and writes use a temp-file rename so a
crash cannot leave a truncated entry that is later served as valid text. PaddleOCR engine
initialisation (~6–16s, once per process) is not avoided by this.

### 2. NL→SQL translation cache — the SQL, never the rows

Caches the **generated SQL**, not the result rows. On a hit the SQL is re-executed against
the live database, so answers always reflect current data. Caching rows would be stale the
moment a document is processed; caching the translation never is.

A hit removes one LLM round-trip (the NL→SQL call) from the request. The answer-phrasing
call is not cached, so it remains on the critical path.

- Key: `SHA-256(normalised_question + schema_version)`
- `schema_version` is a hash of the live `CREATE TABLE` DDL — verified that adding one column
  changed it and **orphaned every entry**, so cached SQL can never reference a dropped column
- Normalisation handles case, whitespace and trailing punctuation **only** — never words that
  carry meaning. `over 500kg` and `under 500kg` keep distinct keys
- SQL is stored only after it executes successfully, so a query the guard rejected is never cached

### 3. Semantic tier — measured and rejected

Full report: **[`docs/SEMANTIC_CACHE.md`](docs/SEMANTIC_CACHE.md)**

A semantic tier was built and swept before being adopted. It was **not shipped**, because the
measurement said not to.

Embedding questions locally with `all-MiniLM-L6-v2`, the two highest-scoring pairs in the set
**both mean the opposite of each other**:

| Cosine | Pair | Meaning |
|---|---|---|
| **0.9855** | `shipments over 500kg` / `shipments under 500kg` | **opposite** |
| **0.9561** | `going to Mumbai` / `coming from Mumbai` | **opposite** |
| 0.9059 | `how many shipments` / `total number of shipments` | same |

No cosine threshold separates the populations. Every threshold that serves a cache hit also
serves a false one; at the most favourable operating point (0.89), **2 of 5 answers served
from the cache would be wrong SQL** for a question the user never asked. The only threshold
with zero false hits serves nothing.

This is structural rather than a tuning problem: a near-miss differing by one word —
`over`/`under`, `loaded`/`discharged`, `passed`/`failed` — is lexically almost identical while
meaning the reverse. The distinction a SQL cache needs is exactly what the embedding discards.

**Decision: ship the exact-match tier only.** It already works, cannot produce a false hit,
and needs no model. The experiment is recorded rather than deleted so the tier is not
proposed again without this evidence. The report states its own provenance limits — the pair
set is adversarially weighted, so its false-hit *rate* says nothing about production traffic;
what survives is the ordering.

---

## Reproducing Every Number

No figure in this README is hand-written. Each one is produced by a script that
regenerates its report, so any claim can be checked or disputed by re-running it.

| Claim | Command | Report |
|---|---|---|
| 150 tests pass, no network | `cd ai-service && pytest tests/ -v` | — |
| Field/decision accuracy, calibration, latency | `python eval/run_eval.py --runs 3` | [`docs/ACCURACY.md`](docs/ACCURACY.md) |
| Per-document LLM cost, per agent | `python eval/run_eval.py --runs 3` | [`docs/COST.md`](docs/COST.md) |
| OCR + SQL cache savings | `python eval/bench_cache.py` | [`docs/CACHE.md`](docs/CACHE.md) |
| Semantic threshold sweep | `python eval/sweep_semantic.py` | [`docs/SEMANTIC_CACHE.md`](docs/SEMANTIC_CACHE.md) |

Running `eval/run_eval.py`, `eval/bench_cache.py` or `eval/sweep_semantic.py` requires a
working `GROQ_API_KEY` in `ai-service/.env` (the sweep does not — it embeds locally). The
test suite requires nothing.

**Wall-clock timings are load-sensitive and should be re-measured on an idle machine.**
`bench_cache.py` records the load average it ran under and refuses to present its own
absolute numbers as citable above a threshold. Correctness results — accuracy, decision
counts, cost, cosine similarities — are deterministic under load and unaffected.

**Two numbers are historical and not regenerable by a current run:** the
pre-optimisation cost baseline of **$0.000988 per document** and the prompt-boilerplate
fractions. Both were measured before the corresponding change and are recorded in the git
history of `docs/COST.md` — `git log -p docs/COST.md` shows the baseline as it stood before
model routing was applied. They are cited here as *before* figures precisely because the
current code no longer produces them.

---

## Known Findings

**[`docs/FINDINGS.md`](docs/FINDINGS.md)** records 8 defects found while adding test coverage.
They are **recorded, not fixed** — except finding 1, which had to be fixed because the service
could not run at all without it. Each entry states the file, line, what was measured, and what
a fix would involve. Where behaviour is wrong, a test pins the *current* behaviour as a
tripwire, so changing it fails loudly.

| # | Finding | Status |
|---|---|---|
| 1 | Pinned model `llama-3.3-70b-versatile` decommissioned by Groq — every call 404'd and the pipeline silently returned null fields | **Fixed** |
| 2 | Reported confidence does not reflect OCR uncertainty — 0.95 on `Aphle Inc.`, 0.85 on `APP-2O26-OO1` | Recorded |
| 3 | `verifyToken` is a no-op — all 9 routes are unauthenticated | Recorded |
| 4 | SQL guard does not stop stacked statements; SQLite's one-statement rule does, incidentally | Recorded |
| 5 | The same guard wrongly rejects read-only CTEs and comment-prefixed queries | Recorded |
| 6 | Fuzzy cutoff lifts unrelated company names over 0.70 on a shared corporate suffix | Recorded |
| 7 | "Nothing evaluated" and "no problems found" produce the same router decision | Recorded |
| 8 | The two text-quality checks measure different lengths — whitespace alone can flip the extraction path | Recorded |

---

## Example Test Cases

The evaluation set is in **[`test-documents/`](test-documents/)** in this repository, with
hand-labelled expected values in [`eval/ground_truth.json`](eval/ground_truth.json).

The narratives below describe the intended behaviour of each case. **Where measurement
disagrees with the narrative, the measured result is noted** — see
[`docs/ACCURACY.md`](docs/ACCURACY.md) for the per-field breakdown.

### 1. `Test-1_Approved(Nike).pdf` (Clean Digital PDF)
- **Document Type**: Commercial Invoice (Digital)
- **Pipeline Result**: ✅ **Auto Approve**
- **How it works**: The pipeline bypassed OCR and instantly extracted the text via the PyMuPDF fast-path. The Extractor Agent pulled all 8 fields with ~95%+ confidence. The Validator Agent matched all fields perfectly against the rules, resulting in an automatic approval with zero human intervention.

### 2. `Test-2_Amendment(Nike).pdf` (Data Mismatch)
- **Document Type**: Commercial Invoice
- **Pipeline Result**: ❌ **Amendment Required**
- **How it works**: The extracted data contained a critical error that violated the compliance rules. The Validator Agent detected a hard mismatch between the expected allowed values and the found value. The Router Agent flagged this as a failure and generated an "Amendment Draft" explicitly detailing what needs to be changed.

### 3. `Test-3_HumanReview(Apple).pdf` (Handwritten/Scanned)
- **Document Type**: Scanned Document
- **Pipeline Result**: ⚠️ **Human Review**
- **How it works**: This file is a low-quality scan. PaddleOCR extracts the text, but misreads `Apple Inc.` as `Aphle Inc.`. The Validator's **fuzzy matching** correctly recognises the similarity and flags that field as *uncertain* (possible OCR misread) rather than a hard mismatch.

> **Measured result: `amendment_required`, not `human_review`.** The fuzzy match on the
> consignee works exactly as described — but OCR also misreads the HS code `847130` as
> `847I3O` (letter I, letter O), and the extractor reports it at **0.80 confidence**, above
> Apple's 0.72 threshold. It is therefore rule-checked, fails the `8471` prefix, and becomes
> a hard mismatch — which outranks the four uncertain fields under the router's precedence.
> **One OCR character flip changes the pipeline's decision.** This is the single decision
> miss in the 2-of-3 score, and it is the basis of
> [finding 2](docs/FINDINGS.md): the gating depends on a self-reported confidence number that
> does not reflect OCR uncertainty.
