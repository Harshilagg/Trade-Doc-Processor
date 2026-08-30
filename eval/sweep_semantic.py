"""
Threshold sweep for the semantic cache tier.

Embeds each labelled question pair locally with all-MiniLM-L6-v2 and sweeps the
cosine threshold, reporting cache-hit rate against false-hit rate at each step.

Labels come from eval/query_pairs.json, which a human fills in. This script never
writes labels and never infers one from a similarity score — that would make the
sweep measure itself.

Model note: the model is sentence-transformers/all-MiniLM-L6-v2, run through
fastembed's ONNX runtime rather than the sentence-transformers package. Same
weights; PyTorch publishes no macOS x86_64 wheels for Python 3.13, so the
sentence-transformers backend cannot be installed on this machine.

Usage:
    python eval/sweep_semantic.py [--pairs eval/query_pairs.json]
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_pairs(path):
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    pairs = data.get("pairs", [])
    unlabelled = [p["id"] for p in pairs if p.get("same_meaning") in ("", None)]
    if unlabelled:
        raise SystemExit(
            f"{len(unlabelled)} pair(s) are unlabelled: {unlabelled}\n"
            f"Fill in same_meaning (true/false) in {path} first. "
            "See eval/README-query-pairs.md — labels must be human-assigned, "
            "never taken from similarity scores."
        )

    for p in pairs:
        if not isinstance(p["same_meaning"], bool):
            raise SystemExit(
                f"Pair {p['id']}: same_meaning must be a JSON boolean "
                f"(true/false), got {p['same_meaning']!r}."
            )
    return data, pairs


def embed_pairs(pairs):
    """Cosine similarity per pair, embedded locally. No API, no network cost."""
    from fastembed import TextEmbedding
    import numpy as np

    questions = []
    for p in pairs:
        questions.extend([p["question_a"], p["question_b"]])

    model = TextEmbedding(MODEL_NAME)
    vectors = np.array(list(model.embed(questions)))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    sims = []
    for i in range(0, len(vectors), 2):
        sims.append(float(vectors[i] @ vectors[i + 1]))
    return sims


def sweep(pairs, sims, steps):
    same = [(p, s) for p, s in zip(pairs, sims) if p["same_meaning"]]
    diff = [(p, s) for p, s in zip(pairs, sims) if not p["same_meaning"]]

    rows = []
    for threshold in steps:
        hits = sum(1 for _, s in same if s >= threshold)
        false_hits = sum(1 for _, s in diff if s >= threshold)
        rows.append(
            {
                "threshold": threshold,
                "hits": hits,
                "same_total": len(same),
                "false_hits": false_hits,
                "diff_total": len(diff),
            }
        )
    return rows, same, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=os.path.join(REPO_ROOT, "eval", "query_pairs.json"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "docs", "SEMANTIC_CACHE.md"))
    args = ap.parse_args()

    if not os.path.exists(args.pairs):
        raise SystemExit(
            f"No labelled pairs at {args.pairs}.\n"
            "Copy eval/query_pairs.template.json to eval/query_pairs.json and "
            "label each pair by hand first."
        )

    data, pairs = load_pairs(args.pairs)
    sims = embed_pairs(pairs)
    steps = [round(0.50 + 0.01 * i, 2) for i in range(51)]  # 0.50 .. 1.00
    rows, same, diff = sweep(pairs, sims, steps)

    # A threshold is safe only if it admits zero false hits.
    safe = [r for r in rows if r["false_hits"] == 0 and r["hits"] > 0]
    best_safe = max(safe, key=lambda r: r["hits"]) if safe else None

    max_diff_sim = max((s for _, s in diff), default=None)
    min_same_sim = min((s for _, s in same), default=None)
    inverted = (
        max_diff_sim is not None
        and min_same_sim is not None
        and max_diff_sim >= min_same_sim
    )

    report = build_report(data, pairs, sims, rows, same, diff, best_safe, inverted,
                          max_diff_sim, min_same_sim, args)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(report)
    print(f"Wrote {args.out}")
    if best_safe:
        print(f"Safest useful threshold: {best_safe['threshold']} "
              f"({best_safe['hits']} of {best_safe['same_total']} true pairs, 0 false hits)")
    else:
        print("No threshold gives any cache hits without also producing a false hit.")


def build_report(data, pairs, sims, rows, same, diff, best_safe, inverted,
                 max_diff_sim, min_same_sim, args):
    from datetime import datetime, timezone

    lines = []
    w = lines.append
    w("# Semantic cache tier — threshold sweep")
    w("")
    w(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
      "by `eval/sweep_semantic.py`.")
    w("")
    w(f"Model: `{MODEL_NAME}`, embedded locally — no API cost, nothing leaves the "
      "machine. Run through fastembed's ONNX runtime rather than the "
      "sentence-transformers package, because PyTorch publishes no macOS x86_64 "
      "wheels for Python 3.13. Same weights.")
    w("")
    w(f"Pairs: {len(pairs)} hand-labelled ({len(same)} same-meaning, "
      f"{len(diff)} different-meaning) by "
      f"{data.get('_labelled_by') or 'one person'}.")
    w("")

    if inverted:
        w("## Verdict: the semantic tier is not safe at any threshold")
        w("")
        w(f"The highest-scoring **different-meaning** pair ({max_diff_sim:.4f}) scores "
          f"**at or above** the lowest-scoring **same-meaning** pair "
          f"({min_same_sim:.4f}). The two populations overlap, so no cosine cutoff "
          "separates them: every threshold low enough to catch the genuine "
          "paraphrases also admits at least one pair that means the opposite.")
        w("")
        w("This is not a tuning problem. Embedding models place sentences with "
          "similar wording close together, and a near-miss pair differing by one "
          "word — `over` vs `under`, `loaded` vs `discharged`, `passed` vs "
          "`failed` — is lexically almost identical while meaning the opposite. "
          "The signal the cache would need is exactly the signal the embedding "
          "discards.")
        w("")
        w("**Recommendation: ship the exact-match tier only.** It already works, "
          "cannot produce a false hit, and needs no embedding model. A false hit "
          "returns confidently wrong SQL for a question the user asked in good "
          "faith — the cost of that is far above the fraction of a cent a cache "
          "hit saves.")
        w("")
    elif best_safe:
        w("## Verdict: a safe threshold exists")
        w("")
        w(f"At **{best_safe['threshold']}**, {best_safe['hits']} of "
          f"{best_safe['same_total']} same-meaning pairs hit, with "
          f"**0 of {best_safe['diff_total']}** different-meaning pairs wrongly "
          "matched.")
        w("")
        w("This is a recommendation, not a decision — the acceptable false-hit "
          "rate is a product judgement. Pick the final value yourself.")
        w("")
    else:
        w("## Verdict: no threshold produces any useful hits safely")
        w("")

    w("## Sweep")
    w("")
    w("| Threshold | Cache hits (same-meaning) | False hits (different-meaning) |")
    w("|---|---|---|")
    prev = None
    for r in rows:
        signature = (r["hits"], r["false_hits"])
        if signature == prev:
            continue  # collapse identical consecutive rows
        prev = signature
        flag = " **<- unsafe**" if r["false_hits"] else ""
        w(f"| {r['threshold']:.2f} | {r['hits']} of {r['same_total']} | "
          f"{r['false_hits']} of {r['diff_total']}{flag} |")
    w("")
    w("Rows where neither count changes are collapsed. Counts, not percentages: "
      f"{len(pairs)} pairs cannot support a rate.")
    w("")

    w("## Every pair, measured")
    w("")
    w("| Pair | Label | Cosine | Questions |")
    w("|---|---|---|---|")
    for p, s in sorted(zip(pairs, sims), key=lambda x: -x[1]):
        label = "same" if p["same_meaning"] else "**different**"
        w(f"| {p['id']} | {label} | {s:.4f} | `{p['question_a']}` / "
          f"`{p['question_b']}` |")
    w("")
    w("Sorted by similarity. Where a **different** row sits above a same row, no "
      "threshold can separate them.")
    w("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
