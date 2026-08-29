# LLM cost baseline

Generated 2026-08-29 18:13 UTC by `eval/run_eval.py`.

**This is the pre-optimisation baseline. No cost reduction has been applied.**

Extraction and NL->SQL model: `openai/gpt-oss-120b`. Router reasoning and query answer: `openai/gpt-oss-20b`. Token counts are taken from each Groq response's `usage` field, never counted locally.

Pricing source: https://console.groq.com/docs/models (fetched 2026-08-29)

- `openai/gpt-oss-120b`: $0.150 per 1M input, $0.60 per 1M output
- `openai/gpt-oss-20b`: $0.075 per 1M input, $0.30 per 1M output

## Cost per document processed

**$0.000838 per document**, averaged over 9 runs (3 documents x 3 runs).

Total for all 9 runs: $0.007544 across 18 LLM calls, 13,416 prompt tokens and 12,507 completion tokens.

### Per agent, per document run

| Agent | Calls/doc | Prompt tokens | Completion tokens | Cost/doc | Share |
|---|---|---|---|---|---|
| `extractor` | 1.0 | 937 | 798 | $0.000619 | 74% |
| `router_reasoning` | 1.0 | 554 | 592 | $0.000219 | 26% |

Prompt and completion token columns are per document run. The validator makes no LLM call at all and so does not appear.

### Per document

| Document | Path | Prompt tokens | Completion tokens | Cost |
|---|---|---|---|---|
| `Test-1_Approved(Nike).pdf` | digital | 1,133 | 1,344 | $0.000797 |
| `Test-2_Amendment(Nike).pdf` | digital | 1,365 | 1,291 | $0.000776 |
| `Test-3_HumanReview(Apple) (1).pdf` | ocr | 1,974 | 1,534 | $0.000942 |

Averaged per run. Cost tracks prompt size, so the OCR document is not necessarily the most expensive despite being by far the slowest.

## Query agent

The Query Agent is reached through `POST /query`, not `/process`, so it contributes nothing to the per-document cost above. Measured separately over 3 probe questions.

| Step | Calls/question | Prompt tokens | Completion tokens | Cost/question |
|---|---|---|---|---|
| `query_answer` | 1.0 | 275 | 116 | $0.000055 |
| `query_sql` | 1.0 | 776 | 157 | $0.000210 |
| **total** | 2.0 | 1,052 | 272 | **$0.000266** |

Probe questions: `How many shipments are there?`, `Which shipments are from Shanghai?`, `What is the average gross weight?`. These measure cost, not answer quality — there is no ground truth for them.

## Method

- Every Groq call goes through the instrumented proxy in `ai-service/utils/llm_metrics.py`, which records the response's `usage` fields, wall-clock latency and computed cost. It forwards to Groq unchanged: no prompt, model or behaviour is altered by instrumentation.
- Cost is `prompt_tokens/1e6 * input_price + completion_tokens/1e6 * output_price`, using the published prices cited above.
- A model with no published price records `None`, never `0`, so an unpriced call cannot silently look free.

Reproduce with `python eval/run_eval.py --runs 3`.
