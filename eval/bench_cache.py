"""
Cache benchmark — measures what the OCR and SQL caches actually save.

Writes docs/CACHE.md. Every number in that report comes from this script, so the
claims in the README are reproducible rather than asserted.

Usage:
    python eval/bench_cache.py
"""
import argparse
import os
import statistics
import sys
import time
from datetime import datetime, timezone

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
AI_SERVICE = os.path.join(REPO_ROOT, "ai-service")
INVOKED_FROM = os.getcwd()
sys.path.insert(0, AI_SERVICE)
os.chdir(AI_SERVICE)

from utils import ocr_cache, sql_cache  # noqa: E402
from services.ocr_service import get_ocr_engine  # noqa: E402
from services.extraction_service import smart_extraction_pipeline  # noqa: E402


LOAD_WARN_THRESHOLD = 8.0


def load_average():
    try:
        return os.getloadavg()[0]
    except (OSError, AttributeError):
        return None


def summarise_times(values):
    """Median and range. Median resists the outliers a loaded machine produces."""
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "values": values,
    }


def bench_ocr(doc_path, repeats):
    """Cold OCR vs warm cache hit, with engine init timed separately."""
    started = time.perf_counter()
    get_ocr_engine()
    engine_init = time.perf_counter() - started

    cold = []
    text_cold = None
    for _ in range(repeats):
        ocr_cache.clear()
        t = time.perf_counter()
        text_cold = smart_extraction_pipeline(doc_path)
        cold.append(time.perf_counter() - t)

    warm, texts = [], []
    for _ in range(repeats):
        t = time.perf_counter()
        texts.append(smart_extraction_pipeline(doc_path))
        warm.append(time.perf_counter() - t)

    return {
        "engine_init": engine_init,
        "cold": cold,
        "warm": warm,
        "identical": all(t == text_cold for t in texts),
        "chars": len(text_cold or ""),
    }


def bench_sql(question, repeats):
    from services.query_agent import run_query

    sql_cache.clear()
    t = time.perf_counter()
    first = run_query(question)
    miss = time.perf_counter() - t

    hits, again = [], first
    for _ in range(repeats):
        t = time.perf_counter()
        again = run_query(question)
        hits.append(time.perf_counter() - t)

    return {
        "miss": miss,
        "hits": hits,
        "sql": first.get("sql_generated", ""),
        "same_sql": first.get("sql_generated") == again.get("sql_generated"),
        "schema_version": sql_cache.schema_version(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--doc", default=os.path.join(
        REPO_ROOT, "test-documents", "Test-3_HumanReview(Apple) (1).pdf"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "CACHE.md"))
    ap.add_argument("--skip-sql", action="store_true",
                    help="skip the SQL benchmark (it makes live LLM calls)")
    args = ap.parse_args()
    args.out = os.path.join(INVOKED_FROM, args.out)

    print(f"OCR benchmark on {os.path.basename(args.doc)} ...", flush=True)
    ocr = bench_ocr(args.doc, args.repeats)

    sql = None
    if not args.skip_sql:
        print("SQL cache benchmark (makes live LLM calls) ...", flush=True)
        sql = bench_sql("How many shipments are there?", args.repeats)

    lines = []
    w = lines.append
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cold_s = summarise_times(ocr["cold"])
    warm_s = summarise_times(ocr["warm"])
    cold_avg = cold_s["median"]
    warm_avg = warm_s["median"]
    saved = cold_avg - warm_avg
    load_now = load_average()
    spread = (cold_s["max"] / cold_s["min"]) if cold_s["min"] else 0

    w("# Cache benchmark")
    w("")
    w(f"Generated {now} by `eval/bench_cache.py`. Every number here is measured by "
      "that script and reproducible with `python eval/bench_cache.py`.")
    w("")
    w(f"1-minute load average during this run: **{load_now:.2f}**"
      if load_now is not None else "System load unavailable.")
    w("")
    if (load_now is not None and load_now > LOAD_WARN_THRESHOLD) or spread > 3:
        w("> **These timings were taken on a loaded machine and should not be cited.**")
        w("> "
          f"Cold-run spread is {spread:.1f}x between fastest and slowest"
          + (f" at load average {load_now:.0f}" if load_now is not None else "")
          + ". Wall-clock timings under contention measure the machine, not the "
            "cache. Re-run on an idle machine before quoting any absolute number.")
        w("> ")
        w("> The **ratio** between cold and warm is far more robust than either "
          "absolute: both are inflated by the same contention, and a cache hit is a "
          "local file read whose cost is dominated by whatever else is running.")
        w("")
    w("## OCR text cache")
    w("")
    w(f"Document: `{os.path.basename(args.doc)}` ({ocr['chars']} chars extracted), "
      f"{args.repeats} runs each.")
    w("")
    w("| | Median | Range |")
    w("|---|---|---|")
    w(f"| Cold OCR (engine already loaded) | **{cold_avg:.2f}s** | "
      f"{cold_s['min']:.2f}–{cold_s['max']:.2f}s |")
    w(f"| Warm (cache hit) | **{warm_avg:.3f}s** | "
      f"{warm_s['min']:.3f}–{warm_s['max']:.3f}s |")
    w(f"| **Saved per repeat** | **{saved:.2f}s — {saved / cold_avg:.1%}** | |")
    w("")
    w("Medians, not means: a single contended run otherwise dominates the average.")
    w("")
    w(f"Cold runs: {', '.join(f'{c:.2f}s' for c in ocr['cold'])}. "
      f"Warm runs: {', '.join(f'{v:.3f}s' for v in ocr['warm'])}.")
    w("")
    w(f"Extracted text identical on every cache hit: **{ocr['identical']}**.")
    w("")
    w(f"PaddleOCR engine initialisation is a further **{ocr['engine_init']:.2f}s**, once "
      "per process. The cache does not avoid it — it is paid on the first OCR call "
      "whether or not the document itself is cached.")
    w("")
    w("The key is a SHA-256 of the document's bytes, so an edited document misses. "
      "Only the extracted text is cached; extraction, validation and routing still run "
      "on every document.")
    w("")

    if sql:
        hit_s = summarise_times(sql["hits"])
        hit_avg = hit_s["median"]
        w("## NL->SQL translation cache")
        w("")
        w(f"Question: `How many shipments are there?`, {args.repeats} repeats.")
        w("")
        w("| | Time |")
        w("|---|---|")
        w(f"| Miss (LLM generates SQL) | **{sql['miss']:.2f}s** |")
        w(f"| Hit (SQL reused, re-executed) | **{hit_avg:.2f}s** |")
        w(f"| **Saved per repeat** | **{sql['miss'] - hit_avg:.2f}s** |")
        w("")
        w("The residual time on a hit is the answer-phrasing LLM call, which is not "
          "cached. Only the NL->SQL translation is.")
        w("")
        w(f"Generated SQL identical across runs: **{sql['same_sql']}**.")
        w("")
        w(f"Schema version at time of run: `{sql['schema_version']}` — a hash of the "
          "live `CREATE TABLE` DDL. Any schema change alters it and orphans every "
          "cached entry, so cached SQL can never reference a dropped column.")
        w("")
        w("**Rows are never cached.** On a hit the SQL is re-executed against the live "
          "database, so the answer always reflects current data. A result cache would "
          "be stale the moment a document is processed.")
        w("")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
