# Multi-Agent Trade Document Processing Platform

An AI-powered platform that automates trade document processing using a **4-agent pipeline architecture**. Upload a Commercial Invoice, Bill of Lading, or Packing List — the system extracts shipment fields via OCR + LLM, validates them against customer-specific rules, routes a compliance decision, and lets you query the data in natural language.

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
- [Tested Sample Documnets](#example-test-cases)

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
| **AI Service** | Python + FastAPI + Groq LLaMA 3.3 70B | 4-agent pipeline: OCR → Extract → Validate → Route → Query |
| **Storage** | SQLite (shipment data) + AWS S3 (raw files) | Zero-infrastructure persistence — SQLite file travels with the service |

---

## Agent Pipeline

The core of the platform is a **sequential multi-agent pipeline** where each agent has a single responsibility:

### 1. Extractor Agent (`services/ai_service.py`)

- **Input**: Raw text from OCR/PDF extraction
- **Output**: 8 structured shipment fields, each with `{value, confidence, source_evidence}`
- **LLM**: Groq LLaMA 3.3 70B with `temperature=0.1` for deterministic extraction
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
| Groq SDK | LLM inference (LLaMA 3.3 70B) |
| PaddleOCR (PP-OCRv3) | OCR for scanned/handwritten documents |
| PyMuPDF (fitz) | Digital PDF text extraction |
| OpenCV | Image preprocessing for OCR |
| SQLite3 | Embedded database for shipment data |
| Boto3 | AWS S3 file downloads |

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
│   │   └── s3_utils.py            # AWS S3 download utility
│   └── shipments.db               # SQLite database (auto-created on startup)
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
| **Smart OCR routing** | Digital PDFs skip OCR entirely (PyMuPDF fast path). Only scanned/image documents hit PaddleOCR. This cuts processing time from ~20s to <2s for digital docs. |
| **Anti-hallucination in Query Agent** | The NL answer LLM receives only actual SQL results — it cannot invent data. Empty results explicitly return "no data found." |

---

## Example Test Cases

You can view the test documents used to evaluate this pipeline here:
**[Link to Test Documents Folder]** *([Add drive link here](https://drive.google.com/drive/folders/1kTdrTSoQYLC-HO0EYP6VfW5DoZG0BOY0?usp=sharing))*

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
- **How it works**: This file was a low-quality scan with handwriting. The PaddleOCR pipeline extracted the text, but the OCR misread "Apple Inc." as "Aphle Inc.". Instead of throwing a hard mismatch, the Validator's **Fuzzy Matching logic** detected that "Aphle" was highly similar to "Apple". It smartly flagged the field as *Uncertain* (possible OCR misread) rather than a strict error, gracefully routing the document to a human operator for a quick manual review.
