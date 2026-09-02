# Cache benchmark

Generated 2026-09-02 09:52 UTC by `eval/bench_cache.py`. Every number here is measured by that script and reproducible with `python eval/bench_cache.py`.

1-minute load average during this run: **222.15**

> **These timings were taken on a loaded machine and should not be cited.**
> Cold-run spread is 3.5x between fastest and slowest at load average 222. Wall-clock timings under contention measure the machine, not the cache. Re-run on an idle machine before quoting any absolute number.
> 
> The **ratio** between cold and warm is far more robust than either absolute: both are inflated by the same contention, and a cache hit is a local file read whose cost is dominated by whatever else is running.

## OCR text cache

Document: `Test-3_HumanReview(Apple) (1).pdf` (188 chars extracted), 3 runs each.

| | Median | Range |
|---|---|---|
| Cold OCR (engine already loaded) | **57.34s** | 17.93–62.33s |
| Warm (cache hit) | **0.905s** | 0.033–1.988s |
| **Saved per repeat** | **56.43s — 98.4%** | |

Medians, not means: a single contended run otherwise dominates the average.

Cold runs: 17.93s, 62.33s, 57.34s. Warm runs: 1.988s, 0.033s, 0.905s.

Extracted text identical on every cache hit: **True**.

PaddleOCR engine initialisation is a further **15.85s**, once per process. The cache does not avoid it — it is paid on the first OCR call whether or not the document itself is cached.

The key is a SHA-256 of the document's bytes, so an edited document misses. Only the extracted text is cached; extraction, validation and routing still run on every document.
