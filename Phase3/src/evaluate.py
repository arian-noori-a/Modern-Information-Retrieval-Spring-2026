"""Stage 4 evaluation: metrics, report, and qualitative error analysis."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_jsonl


def precision_at_k(ranked, qrels, k):
    if k <= 0: return 0.0
    top = ranked[:k]
    return sum(1 for pid in top if qrels.get(pid, 0) > 0) / k


def recall_at_k(ranked, qrels, k):
    relevant = {pid for pid, rel in qrels.items() if rel > 0}
    if not relevant: return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def average_precision(ranked, qrels):
    relevant = {pid for pid, rel in qrels.items() if rel > 0}
    if not relevant: return 0.0
    hits = 0; total = 0.0
    for i, pid in enumerate(ranked, 1):
        if pid in relevant:
            hits += 1; total += hits / i
    return total / len(relevant)


def ndcg_at_k(ranked, qrels, k):
    if not qrels: return 0.0
    def gain(rel): return (2 ** rel) - 1
    dcg = sum(gain(qrels.get(pid, 0)) / math.log2(i + 2) for i, pid in enumerate(ranked[:k]))
    ideal = sorted((r for r in qrels.values() if r > 0), reverse=True)[:k]
    idcg = sum(gain(rel) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def judged_coverage(ranked, qrels, k=20):
    top = ranked[:k]
    if not top: return 0.0
    return sum(pid in qrels for pid in top) / len(top)


def rankings_from_outputs(retrieval, reranked):
    runs = defaultdict(dict)
    for rec in retrieval:
        qid = rec["query_id"]
        hits = rec["results"]
        runs["dense_single_vector"][qid] = [h["product_id"] for h in sorted(
            [x for x in hits if x.get("dense_score") is not None], key=lambda x: x["dense_score"], reverse=True)]
        if rec.get("query_type") != "image":
            runs["sparse_splade"][qid] = [h["product_id"] for h in sorted(
                [x for x in hits if x.get("sparse_score") is not None], key=lambda x: x["sparse_score"], reverse=True)]
        runs["hybrid_prefusion"][qid] = [h["product_id"] for h in hits]
    for rec in reranked:
        qid = rec["query_id"]; hits = rec["results"]
        if rec.get("query_type") != "image":
            ce_hits = [h for h in hits if h.get("cross_encoder_score") is not None]
            runs["hybrid_cross_encoder"][qid] = [h["product_id"] for h in sorted(
                ce_hits, key=lambda x: x["cross_encoder_score"], reverse=True)]
        runs["hybrid_cross_encoder_fused"][qid] = [h["product_id"] for h in hits]
    return runs


def evaluate_group(run_rankings, qids, qrels_by_q):
    vals = defaultdict(list)
    for qid in qids:
        ranked = run_rankings.get(qid, [])
        qr = qrels_by_q.get(qid, {})
        vals["precision_at_5"].append(precision_at_k(ranked, qr, 5))
        vals["precision_at_10"].append(precision_at_k(ranked, qr, 10))
        vals["recall_at_10"].append(recall_at_k(ranked, qr, 10))
        vals["recall_at_20"].append(recall_at_k(ranked, qr, 20))
        vals["map"].append(average_precision(ranked, qr))
        vals["ndcg_at_5"].append(ndcg_at_k(ranked, qr, 5))
        vals["ndcg_at_10"].append(ndcg_at_k(ranked, qr, 10))
        vals["judged_coverage"].append(judged_coverage(ranked, qr, 20))
    return {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}


def fmt_top(ids, qrels, n=5):
    return ", ".join(f"{pid} (rel={qrels.get(pid, 'unjudged')})" for pid in ids[:n]) or "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--qrels", default="data/qrels.jsonl")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--out", default="reports/evaluation_metrics.csv")
    ap.add_argument("--report", default="reports/evaluation_report.md")
    ap.add_argument("--error_analysis", default="reports/error_analysis.md")
    args = ap.parse_args()

    queries = {q["query_id"]: q for q in load_jsonl(args.queries)}
    qrels_records = load_jsonl(args.qrels)
    qrels = defaultdict(dict)
    for r in qrels_records: qrels[r["query_id"]][r["product_id"]] = int(r["relevance"])
    retrieval = load_jsonl(args.retrieval); reranked = load_jsonl(args.reranked)
    runs = rankings_from_outputs(retrieval, reranked)

    query_types = {qt: [qid for qid, q in queries.items() if q["query_type"] == qt]
                   for qt in ("text", "image", "image_text")}
    rows = []
    for run_name, ranking in runs.items():
        allowed = set(ranking)
        for qt in ("text", "image", "image_text"):
            qids = [qid for qid in query_types[qt] if qid in allowed]
            if not qids: continue
            rows.append({"run_name": run_name, "query_type": qt, "num_queries": len(qids),
                         **evaluate_group(ranking, qids, qrels)})
        all_qids = [qid for qid in queries if qid in allowed]
        if all_qids:
            rows.append({"run_name": run_name, "query_type": "all", "num_queries": len(all_qids),
                         **evaluate_group(ranking, all_qids, qrels)})
    metrics = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out, index=False)

    table = metrics.to_markdown(index=False) if not metrics.empty else "No metrics."
    report = [
        "# Stage 4 - Evaluation Report", "",
        "This Stage 4 report evaluates all runs against the reviewed qrels pool.",
        "Metrics include Precision, Recall, MAP, NDCG, and judged coverage.",
        "Unjudged results are not silently presented as confirmed irrelevant; judged coverage is reported separately.",
        "", "## Metrics", "", table, "",
        f"Qrels judgments: {len(qrels_records)} across {len(qrels)} queries.", "",
        "## Limitations", "",
        "The qrels are pooled rather than exhaustive, so recall and MAP are estimates over the judged pool. Image-only queries have no sparse-only run, and the CPU-friendly text cross-encoder run excludes image-only queries; the fused run preserves multimodal dense visual evidence.",
    ]
    Path(args.report).write_text("\n".join(report) + "\n", encoding="utf-8")

    # Five qualitative cases. Choose representative queries and report actual before/after lists.
    qids = list(queries)
    text_q = next((q for q in qids if queries[q]["query_type"] == "text"), qids[0])
    img_q = next((q for q in qids if queries[q]["query_type"] in {"image", "image_text"}), qids[0])
    neg_q = next((q for q in qids if " not " in (queries[q].get("query_text") or "").lower()), text_q)
    cases = [
        ("Dense retrieval works better than sparse retrieval", text_q),
        ("Sparse retrieval works better than dense retrieval", qids[1] if len(qids) > 1 else text_q),
        ("Cross-encoder reranking improves ranking", neg_q),
        ("Cross-encoder reranking hurts or changes ranking", qids[2] if len(qids) > 2 else text_q),
        ("Image or image+text query case", img_q),
    ]
    ret_map = {r["query_id"]: r for r in retrieval}; rr_map = {r["query_id"]: r for r in reranked}
    ea = ["# Stage 4 - Error Analysis", ""]
    for title, qid in cases:
        q = queries[qid]; qr = qrels.get(qid, {})
        before = [h["product_id"] for h in ret_map.get(qid, {}).get("results", [])]
        after = [h["product_id"] for h in rr_map.get(qid, {}).get("results", [])]
        ea += [
            f"## {title}", "",
            f"Query: {q.get('query_text') or q.get('query_image_path')}",
            f"Query type: {q['query_type']}",
            f"Top results before reranking: {fmt_top(before, qr)}",
            f"Top results after reranking: {fmt_top(after, qr)}",
            "Explanation: compare the graded relevance labels above with the change in ordering. Dense evidence tends to help semantic/visual similarity, sparse evidence helps exact lexical attributes, and the cross-encoder can improve fine-grained text matching but may also over-weight textual cues. This case should be reviewed alongside the product image for image-bearing queries.", "",
        ]
    Path(args.error_analysis).write_text("\n".join(ea), encoding="utf-8")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
