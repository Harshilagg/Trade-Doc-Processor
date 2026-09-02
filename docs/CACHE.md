# Cache benchmark

Generated 2026-09-02 10:12 UTC by `eval/bench_cache.py`. Every number here is measured by that script and reproducible with `python eval/bench_cache.py`.

1-minute load average during this run: **65.65**

> **These timings were taken on a loaded machine and should not be cited.**
> Cold-run spread is 2.1x between fastest and slowest at load average 66. Wall-clock timings under contention measure the machine, not the cache. Re-run on an idle machine before quoting any absolute number.
> 
> The **ratio** between cold and warm is far more robust than either absolute: both are inflated by the same contention, and a cache hit is a local file read whose cost is dominated by whatever else is running.

## OCR text cache

Document: `Test-3_HumanReview(Apple) (1).pdf` (188 chars extracted), 5 runs each.

| | Median | Range |
|---|---|---|
| Cold OCR (engine already loaded) | **23.75s** | 17.48–36.04s |
| Warm (cache hit) | **0.027s** | 0.018–0.147s |
| **Saved per repeat** | **23.73s — 99.9%** | |

Medians, not means: a single contended run otherwise dominates the average.

Cold runs: 23.75s, 36.04s, 21.56s, 30.60s, 17.48s. Warm runs: 0.147s, 0.030s, 0.027s, 0.018s, 0.027s.

Extracted text identical on every cache hit: **True**.

PaddleOCR engine initialisation is a further **9.66s**, once per process. The cache does not avoid it — it is paid on the first OCR call whether or not the document itself is cached.

The key is a SHA-256 of the document's bytes, so an edited document misses. Only the extracted text is cached; extraction, validation and routing still run on every document.

## NL->SQL translation cache

Question: `How many shipments are there?`, 5 repeats.

| | Time |
|---|---|
| Miss (LLM generates SQL) | **1.47s** |
| Hit (SQL reused, re-executed) | **0.46s** |
| **Saved per repeat** | **1.01s** |

The residual time on a hit is the answer-phrasing LLM call, which is not cached. Only the NL->SQL translation is.

Generated SQL identical across runs: **True**.

Schema version at time of run: `b83a8eb5931eff05` — a hash of the live `CREATE TABLE` DDL. Any schema change alters it and orphans every cached entry, so cached SQL can never reference a dropped column.

**Rows are never cached.** On a hit the SQL is re-executed against the live database, so the answer always reflects current data. A result cache would be stale the moment a document is processed.
