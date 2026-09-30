"""Build a pooled qrels worksheet and finalize reviewed judgments."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import load_jsonl, write_jsonl


def top_by_score(results, field, n):
    rows = [h for h in results if h.get(field) is not None]
    rows.sort(key=lambda h: float(h[field]), reverse=True)
    return [h["product_id"] for h in rows[:n]]


def build_pool(catalog_path, queries_path, retrieval_path, reranked_path, out_path, per_run):
    catalog = pd.read_parquet(catalog_path)
    cat = catalog.set_index("product_id", drop=False)
    queries = {q["query_id"]: q for q in load_jsonl(queries_path)}
    retrieval = {r["query_id"]: r for r in load_jsonl(retrieval_path)}
    reranked = {r["query_id"]: r for r in load_jsonl(reranked_path)}

    rows = []
    for qid, q in queries.items():
        ret = retrieval.get(qid, {"results": []})["results"]
        rr = reranked.get(qid, {"results": []})["results"]
        systems = {
            "dense_single_vector": top_by_score(ret, "dense_score", per_run),
            "sparse_splade": top_by_score(ret, "sparse_score", per_run) if q["query_type"] != "image" else [],
            "hybrid_prefusion": [h["product_id"] for h in ret[:per_run]],
            "hybrid_cross_encoder_fused": [h["product_id"] for h in rr[:per_run]],
        }
        pooled = []
        seen = set()
        source_map = {}
        for system, ids in systems.items():
            for pid in ids:
                source_map.setdefault(pid, []).append(system)
                if pid not in seen:
                    seen.add(pid); pooled.append(pid)
        for pid in pooled:
            if pid not in cat.index:
                continue
            p = cat.loc[pid]
            rows.append({
                "query_id": qid,
                "query_type": q["query_type"],
                "query_text": q.get("query_text"),
                "query_image_path": q.get("query_image_path"),
                "product_id": pid,
                "pool_sources": ",".join(source_map.get(pid, [])),
                "title": p.get("title", ""),
                "product_type": p.get("product_type", ""),
                "brand": p.get("brand", ""),
                "color": p.get("color", ""),
                "material": p.get("material", ""),
                "style": p.get("style", ""),
                "image_path": p.get("image_path", ""),
                "product_text": p.get("product_text", ""),
                "relevance": "",
                "reason": "",
            })
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} pooled rows to {out_path}. Review relevance (0/1/2) and reason manually.")


def finalize(pool_path, qrels_out):
    df = pd.read_csv(pool_path, keep_default_na=False)
    if "relevance" not in df or "reason" not in df:
        raise ValueError("Pool must contain relevance and reason columns.")
    bad_rel = []
    records = []
    for i, row in df.iterrows():
        try:
            rel = int(row["relevance"])
        except Exception:
            bad_rel.append(i + 2); continue
        if rel not in {0, 1, 2} or not str(row["reason"]).strip():
            bad_rel.append(i + 2); continue
        records.append({
            "query_id": str(row["query_id"]),
            "product_id": str(row["product_id"]),
            "relevance": rel,
            "reason": str(row["reason"]).strip(),
        })
    if bad_rel:
        raise ValueError(
            f"Rows with missing/invalid relevance or reason: {bad_rel[:20]}"
            + (" ..." if len(bad_rel) > 20 else "")
        )
    write_jsonl(qrels_out, records)
    print(f"Wrote {len(records)} reviewed qrels to {qrels_out}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--retrieval", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--reranked", default="outputs/reranked_results.jsonl")
    ap.add_argument("--pool_out", default="data/qrels_pool.csv")
    ap.add_argument("--per_run", type=int, default=10)
    ap.add_argument("--finalize")
    ap.add_argument("--qrels_out", default="data/qrels.jsonl")
    args = ap.parse_args()
    if args.finalize:
        finalize(args.finalize, args.qrels_out)
    else:
        build_pool(args.catalog, args.queries, args.retrieval, args.reranked, args.pool_out, args.per_run)


if __name__ == "__main__":
    main()
