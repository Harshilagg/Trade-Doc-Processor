# Query pairs — labelling for the semantic cache tier

## The rule

**You label these, not the embedding model.**

Each pair gets `same_meaning: true` or `false`, decided by reading the two
questions and asking: *should these two produce the same SQL?* Never by looking at
a similarity score.

The whole point of the threshold sweep is to test whether cosine similarity agrees
with human judgement. If the labels come from similarity in the first place, the
sweep measures nothing and will report a perfect threshold that does not exist.

The template ships with every `same_meaning` empty for that reason.

## How to fill it in

1. Copy the template so it stays regenerable:

   ```
   cp eval/query_pairs.template.json eval/query_pairs.json
   ```

2. For each pair set `same_meaning` to `true` or `false` (JSON booleans, not
   strings).

3. Fill in `_labelled_by` and `_labelled_date`.

4. Add your own pairs. Fifteen is a starting point, not a target — the sweep gets
   sharper with more, especially more near-misses.

## What the labels mean

| Value | Meaning | Consequence in the cache |
|---|---|---|
| `true` | Both questions should return the **same SQL** | A cache hit here is correct and saves an LLM call |
| `false` | They must **not** share a cache entry | A cache hit here is a **false hit** — the user gets confidently wrong SQL |

Judge on the SQL, not on topic. "How many shipments does Nike have?" and "How many
shipments does Apple have?" are about the same subject and read almost identically,
but they need different `WHERE` clauses, so they are `false`.

If a pair genuinely sits between the two, write why in `notes` and pick the answer
that would be safer to get wrong. A missed cache hit costs a fraction of a cent; a
false hit returns an answer that looks authoritative and is wrong.

## Why the near-misses matter most

Pairs like these are the reason the sweep exists:

- `shipments over 500kg` / `shipments under 500kg`
- `which shipments loaded at Shanghai?` / `which shipments discharged at Shanghai?`
- `how many documents failed validation?` / `how many documents passed validation?`

Each pair differs by one word, embeds almost identically, and means the **opposite**.
Embedding models are trained to place similar-sounding sentences close together, and
these are as close as sentences get while meaning opposite things. They are exactly
where a semantic cache breaks.

A threshold that catches the genuine paraphrases while rejecting all of these is
what makes the tier shippable. If no such threshold exists, the honest outcome is to
ship only the exact-match tier — which already works and cannot produce a false hit.

## What happens next

Once `eval/query_pairs.json` is filled in, the sweep will:

1. Embed every question locally with `sentence-transformers` /
   `all-MiniLM-L6-v2` — no API cost, nothing leaves the machine.
2. Compute cosine similarity for each pair.
3. Sweep the threshold across its range, and at each step report **cache-hit rate**
   (true pairs correctly matched) against **false-hit rate** (false pairs wrongly
   matched).
4. Report the table and recommend a threshold.

**You pick the final value.** The recommendation is a reading of the measured
trade-off, not a decision — the cost of a false hit is a product judgement, not a
statistical one.
