"""Stage 3 CPU-friendly cross-encoder reranking with multimodal dense evidence fusion."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from common import load_jsonl, product_rerank_text, write_jsonl, zscore

DEFAULT_WEIGHTS = {
    "text": {"ce": 0.65, "dense": 0.20, "sparse": 0.15},
    "image": {"ce": 0.00, "dense": 1.00, "sparse": 0.00},
    "image_text": {"ce": 0.55, "dense": 0.30, "sparse": 0.15},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rerank_top_n", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_chars", type=int, default=3000)
    ap.add_argument("--out", default="outputs/reranked_results.jsonl")
    ap.add_argument("--strategy_a_out", default="outputs/reranked_strategy_a.jsonl")
    ap.add_argument("--summary", default="reports/reranking_summary.md")
    args = ap.parse_args()

    from sentence_transformers import CrossEncoder

    catalog = pd.read_parquet(args.catalog)
    by_id = {r.product_id: r for r in catalog.itertuples()}
    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    retrieval = load_jsonl(args.retrieval)
    ce = CrossEncoder(args.reranker, device=args.device, max_length=512)

    fused_records, ce_only_records = [], []
    before_after = []
    for rec in tqdm(retrieval, desc="rerank"):
        q = queries[rec["query_id"]]
        qt = q["query_type"]
        hits = rec["results"][:args.rerank_top_n]
        ce_scores: list[float | None]
        if qt == "image":
            # A text cross-encoder has no visual input; retain visual evidence through
            # the required dense multimodal score in the fused ranking.
            ce_scores = [None] * len(hits)
        else:
            pairs = [
                (q["query_text"], product_rerank_text(by_id[h["product_id"]], args.max_chars))
                for h in hits
            ]
            pred = ce.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            ce_scores = [float(x) for x in np.asarray(pred).reshape(-1)]

        ce_z = zscore(ce_scores) if qt != "image" else [0.0] * len(hits)
        d_z = zscore([h.get("dense_score") for h in hits])
        s_z = zscore([h.get("sparse_score") for h in hits]) if qt != "image" else [0.0] * len(hits)
        w = DEFAULT_WEIGHTS[qt]
        rows = []
        for i, h in enumerate(hits):
            final = w["ce"] * ce_z[i] + w["dense"] * d_z[i] + w["sparse"] * s_z[i]
            row = dict(h)
            row["cross_encoder_score"] = ce_scores[i]
            row["final_score"] = float(final)
            rows.append(row)
        rows.sort(key=lambda h: h["final_score"], reverse=True)
        for rank, h in enumerate(rows, 1): h["rank"] = rank
        fused_records.append({
            "query_id": rec["query_id"], "query_type": qt,
            "run_name": "hybrid_cross_encoder_fusion",
            "reranker_type": "text_cross_encoder_plus_multimodal_dense_fusion",
            "results": rows,
        })

        if qt != "image":
            a_rows = sorted((dict(h) for h in rows), key=lambda h: h["cross_encoder_score"], reverse=True)
            for rank, h in enumerate(a_rows, 1):
                h["rank"] = rank; h["final_score"] = float(h["cross_encoder_score"])
            ce_only_records.append({
                "query_id": rec["query_id"], "query_type": qt,
                "run_name": "hybrid_cross_encoder",
                "reranker_type": "text_cross_encoder_only",
                "results": a_rows,
            })
        before_after.append((rec, fused_records[-1]))

    write_jsonl(args.out, fused_records)
    write_jsonl(args.strategy_a_out, ce_only_records)

    examples = []
    selected = before_after[:3]
    neg = next((x for x in before_after if " not " in (queries[x[0]["query_id"]].get("query_text") or "").lower()), None)
    imgtext = next((x for x in before_after if x[0]["query_type"] == "image_text"), None)
    for x in [neg, imgtext]:
        if x and x not in selected: selected.append(x)
    for before, after in selected[:5]:
        b = [h["product_id"] for h in before["results"][:3]]
        a = [h["product_id"] for h in after["results"][:3]]
        examples.append(f"- **{before['query_id']}** ({before['query_type']}): before={b}; after={a}")
    lines = [
        "# Stage 3 - Reranking Summary", "",
        f"Reranker model: `{args.reranker}`.",
        "Reranker input format: `(query_text, compact product metadata text)` for text-bearing queries.",
        "Image-only queries are not sent to the text cross-encoder; their final ordering retains the multimodal dense visual score.",
        f"Reranking depth: {args.rerank_top_n}.",
        "Score normalization method: per-query z-score.",
        f"Fusion weights: {DEFAULT_WEIGHTS}.",
        f"Runtime environment/device: {args.device}.", "",
        "This CPU-friendly main fused run preserves visual evidence for image and image+text queries through the Stage 1/2 multimodal dense score.",
        "", "## Before/after examples", "", *examples,
    ]
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
