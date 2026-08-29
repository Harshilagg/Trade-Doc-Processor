import os

# python-dotenv is optional: present in venv/dev, may be absent in some prod containers.
# os.environ vars are always read regardless.
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────
# ARCHITECTURE NOTE:
# Firebase vars removed from REQUIRED_VARS because SQLite is the
# primary shipment data store now. Firebase is still used for AUTH
# only (token verification in server.js). The Python AI service
# no longer needs firebase-admin at all.
# ─────────────────────────────────────────────────────────────────
REQUIRED_VARS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    "AWS_BUCKET_NAME",
    "GROQ_API_KEY"
]

def validate_config():
    missing = [var for var in REQUIRED_VARS if not os.getenv(var)]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

class Config:
    # AWS — kept for S3 document storage (raw files remain in S3)
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = os.getenv("AWS_REGION")
    AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
    
    # Groq — used by: Extractor Agent, Router Agent, Query Agent
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    # Was "llama-3.3-70b-versatile", which Groq has decommissioned — it returns
    # 404 model_not_found, and no Llama chat model is available. See
    # docs/FINDINGS.md finding 1. This is the model docs/ACCURACY.md's baseline
    # was measured on, so config and reported numbers describe the same model.
    GROQ_MODEL = "openai/gpt-oss-120b"

    # SQLite — replaces Firestore for shipment/pipeline data persistence
    # ARCHITECTURE: SQLite chosen for zero-infra local storage; portable for demo
    SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "./shipments.db")

    # Customer rules file path
    CUSTOMER_RULES_PATH = os.getenv("CUSTOMER_RULES_PATH", "./customer_rules.json")

    # Confidence threshold below which a field is marked "uncertain"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.60

    # Agent retry config — prevents infinite loops
    MAX_LLM_RETRIES = 2
    LLM_TIMEOUT_SECONDS = 30.0


# PaddleOCR optimization flags — critical for CPU stability
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
