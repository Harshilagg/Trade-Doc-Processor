"""
OCR result cache — keyed on document bytes.

OCR is by far the slowest step in the pipeline (10-22s per scan, against 0.01s
for a digital PDF, docs/ACCURACY.md). Re-processing the same document repeats
that cost for a result that cannot have changed: the key is a SHA-256 of the
file's bytes, so any edit to the document is a different key.

This caches only the extracted *text*. Nothing downstream is cached — the same
text still goes through extraction, validation and routing every time.
"""
import hashlib
import os

from config import Config
from logger import logger

_stats = {"hits": 0, "misses": 0, "writes": 0}


def reset_stats():
    for key in _stats:
        _stats[key] = 0


def get_stats():
    return dict(_stats)


def hash_file(file_path):
    """SHA-256 of the file's bytes, read in chunks so large scans stay cheap."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_path(file_hash):
    return os.path.join(Config.OCR_CACHE_DIR, f"{file_hash}.txt")


def get(file_path):
    """Cached OCR text for this document, or None on a miss."""
    if not Config.OCR_CACHE_ENABLED:
        return None
    try:
        path = _cache_path(hash_file(file_path))
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            _stats["hits"] += 1
            logger.info(
                f"[OCRCache] HIT {os.path.basename(file_path)} "
                f"({len(text)} chars) — skipping OCR"
            )
            return text
    except Exception as exc:
        # A cache failure must never break extraction: fall through to real OCR.
        logger.warning(f"[OCRCache] read failed, falling back to OCR: {exc}")
        return None

    _stats["misses"] += 1
    return None


def put(file_path, text):
    """Store OCR text for this document. Failures are logged, never raised."""
    if not Config.OCR_CACHE_ENABLED or text is None:
        return
    try:
        os.makedirs(Config.OCR_CACHE_DIR, exist_ok=True)
        path = _cache_path(hash_file(file_path))
        # Write to a temp file then rename, so a crash mid-write cannot leave a
        # truncated entry that would later be served as a valid result.
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
        _stats["writes"] += 1
        logger.info(f"[OCRCache] stored {len(text)} chars for {os.path.basename(file_path)}")
    except Exception as exc:
        logger.warning(f"[OCRCache] write failed (non-fatal): {exc}")


def clear():
    """Remove every cached entry. Used by tests and by --no-cache eval runs."""
    directory = Config.OCR_CACHE_DIR
    if not os.path.isdir(directory):
        return 0
    removed = 0
    for name in os.listdir(directory):
        if name.endswith(".txt") or name.endswith(".tmp"):
            os.remove(os.path.join(directory, name))
            removed += 1
    return removed
