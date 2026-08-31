# Semantic cache tier — threshold sweep

Generated 2026-08-31 08:40 UTC by `eval/sweep_semantic.py`.

Model: `sentence-transformers/all-MiniLM-L6-v2`, embedded locally — no API cost, nothing leaves the machine. Run through fastembed's ONNX runtime rather than the sentence-transformers package, because PyTorch publishes no macOS x86_64 wheels for Python 3.13. Same weights.

Pairs: 15 hand-labelled (6 same-meaning, 9 different-meaning) by Claude (assistant) - NOT independently labelled by the repository owner. See docs/SEMANTIC_CACHE.md 'Provenance' for why this weakens the result..

## Verdict: the semantic tier is not safe at any threshold

The highest-scoring **different-meaning** pair (0.9855) scores **at or above** the lowest-scoring **same-meaning** pair (0.6041). The two populations overlap, so no cosine cutoff separates them: every threshold low enough to catch the genuine paraphrases also admits at least one pair that means the opposite.

This is not a tuning problem. Embedding models place sentences with similar wording close together, and a near-miss pair differing by one word — `over` vs `under`, `loaded` vs `discharged`, `passed` vs `failed` — is lexically almost identical while meaning the opposite. The signal the cache would need is exactly the signal the embedding discards.

**Recommendation: ship the exact-match tier only.** It already works, cannot produce a false hit, and needs no embedding model. A false hit returns confidently wrong SQL for a question the user asked in good faith — the cost of that is far above the fraction of a cent a cache hit saves.

## Sweep

`Served` is every pair the cache would treat as a hit at that threshold. `Wrong` is how many of those return SQL for a different question — the number that actually reaches a user.

| Threshold | Cache hits (same-meaning) | False hits (different-meaning) | Served | Wrong |
|---|---|---|---|---|
| 0.50 | 6 of 6 | 9 of 9 **<- unsafe** | 15 | **9 of 15** |
| 0.54 | 6 of 6 | 8 of 9 **<- unsafe** | 14 | **8 of 14** |
| 0.61 | 5 of 6 | 8 of 9 **<- unsafe** | 13 | **8 of 13** |
| 0.68 | 5 of 6 | 7 of 9 **<- unsafe** | 12 | **7 of 12** |
| 0.74 | 4 of 6 | 7 of 9 **<- unsafe** | 11 | **7 of 11** |
| 0.77 | 4 of 6 | 6 of 9 **<- unsafe** | 10 | **6 of 10** |
| 0.79 | 3 of 6 | 5 of 9 **<- unsafe** | 8 | **5 of 8** |
| 0.81 | 3 of 6 | 4 of 9 **<- unsafe** | 7 | **4 of 7** |
| 0.86 | 3 of 6 | 3 of 9 **<- unsafe** | 6 | **3 of 6** |
| 0.89 | 3 of 6 | 2 of 9 **<- unsafe** | 5 | **2 of 5** |
| 0.90 | 1 of 6 | 2 of 9 **<- unsafe** | 3 | **2 of 3** |
| 0.91 | 0 of 6 | 2 of 9 **<- unsafe** | 2 | **2 of 2** |
| 0.96 | 0 of 6 | 1 of 9 **<- unsafe** | 1 | **1 of 1** |
| 0.99 | 0 of 6 | 0 of 9 | 0 | — |

Rows where neither count changes are collapsed. Counts, not percentages: 15 pairs cannot support a rate.

### Best available operating point

The most favourable threshold that serves anything is **0.89**: 3 correct hits and **2 false hits**, so **2 of 5 answers served from the cache would be wrong**.

That is the best case, on a set chosen by the same person who wrote the tool. It is not a shippable operating point for a system that answers questions about shipping compliance.

## Every pair, measured

| Pair | Label | Cosine | Questions |
|---|---|---|---|
| 2 | **different** | 0.9855 | `Show me shipments over 500kg` / `Show me shipments under 500kg` |
| 5 | **different** | 0.9561 | `Which shipments are going to Mumbai?` / `Which shipments are coming from Mumbai?` |
| 1 | same | 0.9059 | `How many shipments are there?` / `What is the total number of shipments?` |
| 4 | same | 0.8947 | `List shipments for Nike` / `Show all Nike shipments` |
| 6 | same | 0.8927 | `What is the average gross weight?` / `What is the mean gross weight?` |
| 9 | **different** | 0.8844 | `How many documents failed validation?` / `How many documents passed validation?` |
| 11 | **different** | 0.8540 | `Which shipments loaded at Shanghai?` / `Which shipments discharged at Shanghai?` |
| 7 | **different** | 0.8026 | `What is the average gross weight?` / `What is the maximum gross weight?` |
| 8 | **different** | 0.7885 | `Show shipments with incoterms FOB` / `Show shipments with incoterms CIF` |
| 14 | same | 0.7849 | `What HS codes are in the system?` / `List the distinct HS codes` |
| 12 | **different** | 0.7616 | `Show me the most recent shipment` / `Show me the oldest shipment` |
| 10 | same | 0.7352 | `List all shipments` / `Show me every shipment` |
| 13 | **different** | 0.6702 | `How many shipments does Nike have?` / `How many shipments does Apple have?` |
| 15 | same | 0.6041 | `Show shipments requiring amendment` / `Show shipments that need to be corrected` |
| 3 | **different** | 0.5377 | `Which shipments were auto approved?` / `Which shipments need human review?` |

Sorted by similarity. Where a **different** row sits above a same row, no threshold can separate them.

## Provenance — read before citing this

Labelled by: **Claude (assistant) - NOT independently labelled by the repository owner. See docs/SEMANTIC_CACHE.md 'Provenance' for why this weakens the result.**

Three limits on this result, stated so it is not over-claimed:

1. **The labels and the pairs come from the same source that built the tool.** Independent labelling by someone who had not seen the implementation would be stronger evidence. The labels were decided on SQL grounds — which `WHERE` clause or aggregate each question needs — and each pair's reasoning is recorded in `eval/query_pairs.json`, so they can be checked and disagreed with.

2. **The pair set is adversarially weighted.** It was written to probe near-misses, so the proportion of different-meaning pairs is a design choice, not a sample of real traffic. **The false-hit *rate* therefore means nothing about production**, and no rate should be quoted from it.

3. **What does survive those limits is the ordering.** Whether two questions need the same SQL is not a matter of taste: `over 500kg` and `under 500kg` need opposite comparison operators. That the two highest-scoring pairs in the set both mean opposite things is a property of the embedding, not of the sampling. Rebalancing the set would move every count in the sweep, and would not reorder that table.

A stronger version of this experiment would use real logged questions, labelled by someone other than the tool's author. That is worth doing before the tier is ever reconsidered — but it is unlikely to reverse the conclusion, because the failure is structural rather than statistical.
