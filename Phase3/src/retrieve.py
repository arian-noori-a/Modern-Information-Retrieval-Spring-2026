"""Stage 2 hybrid dense + SPLADE retrieval."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse as sp
from tqdm.auto import tqdm

from common import load_jsonl, minmax_map, resolve_path, write_jsonl
from build_index import encode_dense, load_dense_model, splade_batch


def dense_search(index, q: np.ndarray, top_k: int):
    q = np.asarray(q, dtype=np.float32).reshape(1, -1)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    scores, ids = index.search(np.ascontiguousarray(q), top_k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def sparse_search(matrix: sp.csr_matrix, idx: np.ndarray, val: np.ndarray, top_k: int):
    q = sp.csr_matrix((val, (np.zeros(len(idx), dtype=int), idx)), shape=(1, matrix.shape[1]))
    scores = (matrix @ q.T).toarray().ravel()
    k = min(top_k, len(scores))
    if k <= 0:
        return []
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return [(int(i), float(scores[i])) for i in top]


def merge_hits(d_hits, s_hits, query_type: str, method: str, rrf_k: int,
               w_dense: float, w_sparse: float):
    sparse_ok = query_type != "image"
    d_raw, s_raw = dict(d_hits), dict(s_hits)
    d_norm, s_norm = minmax_map(d_hits), minmax_map(s_hits)
    d_rank = {i: r + 1 for r, (i, _) in enumerate(d_hits)}
    s_rank = {i: r + 1 for r, (i, _) in enumerate(s_hits)}
    rows = []
    for i in set(d_raw) | set(s_raw):
        if method == "rrf":
            score = (1 / (rrf_k + d_rank[i]) if i in d_rank else 0.0)
            if sparse_ok and i in s_rank:
                score += 1 / (rrf_k + s_rank[i])
        else:
            if sparse_ok:
                score = w_dense * d_norm.get(i, 0.0) + w_sparse * s_norm.get(i, 0.0)
            else:
                score = d_norm.get(i, 0.0)
        src = []
        if i in d_raw: src.append("dense")
        if i in s_raw: src.append("sparse")
        rows.append((i, d_raw.get(i), s_raw.get(i) if sparse_ok else None, float(score), src))
    rows.sort(key=lambda x: x[3], reverse=True)
    return rows


def validate_queries(queries):
    for q in queries:
        qt = q.get("query_type")
        if qt not in {"text", "image", "image_text"}:
            raise ValueError(f"Bad query_type for {q.get('query_id')}: {qt}")
        if qt in {"text", "image_text"} and not q.get("query_text"):
            raise ValueError(f"{q['query_id']} requires query_text")
        if qt in {"image", "image_text"} and not q.get("query_image_path"):
            raise ValueError(f"{q['query_id']} requires query_image_path")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--queries", default="data/queries.jsonl")
    ap.add_argument("--index_dir", default="artifacts/index")
    ap.add_argument("--emb_dir", default="artifacts/embeddings")
    ap.add_argument("--top_k_dense", type=int, default=100)
    ap.add_argument("--top_k_sparse", type=int, default=100)
    ap.add_argument("--prefusion", choices=["weighted", "rrf"], default="weighted")
    ap.add_argument("--w_dense", type=float, default=0.55)
    ap.add_argument("--w_sparse", type=float, default=0.45)
    ap.add_argument("--rrf_k", type=int, default=60)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="outputs/retrieval_results.jsonl")
    ap.add_argument("--summary", default="reports/retrieval_summary.md")
    args = ap.parse_args()

    import gc
    import torch, faiss
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    manifest = json.loads(Path(args.index_dir, "index_manifest.json").read_text(encoding="utf-8"))
    product_ids = Path(args.emb_dir, "product_ids.txt").read_text(encoding="utf-8").split()
    dense_index = faiss.read_index(str(Path(args.index_dir, "dense_index.faiss")))
    sparse_matrix = sp.load_npz(Path(args.index_dir, "sparse_index.npz"))
    queries = load_jsonl(args.queries)
    validate_queries(queries)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    root = Path(".").resolve()

    # Phase A: dense query encoding/search. Unload before SPLADE to lower CPU RAM.
    dense_model, _ = load_dense_model(manifest["dense_model"], device)
    dense_hits_by_qid = {}
    for q in tqdm(queries, desc="dense queries"):
        qt = q["query_type"]
        if qt == "text":
            dense_input = q["query_text"]
        elif qt == "image":
            image_path = resolve_path(q["query_image_path"], root)
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing query image: {q['query_image_path']}")
            dense_input = image_path
        else:
            image_path = resolve_path(q["query_image_path"], root)
            if not image_path or not os.path.exists(image_path):
                raise FileNotFoundError(f"Missing query image: {q['query_image_path']}")
            dense_input = {"text": q["query_text"], "image": image_path}
        q_dense = encode_dense(dense_model, [dense_input], batch_size=1, normalize=True, task="query")[0]
        dense_hits_by_qid[q["query_id"]] = dense_search(dense_index, q_dense, args.top_k_dense)
    del dense_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Phase B: sparse text-query encoding/search.
    sp_tokenizer = AutoTokenizer.from_pretrained(manifest["sparse_model"])
    sp_model = AutoModelForMaskedLM.from_pretrained(manifest["sparse_model"]).to(device).eval()
    sparse_hits_by_qid = {}
    for q in tqdm(queries, desc="sparse queries"):
        if q["query_type"] == "image":
            sparse_hits_by_qid[q["query_id"]] = []
            continue
        idx, val = splade_batch(
            sp_model, sp_tokenizer, [q["query_text"]], device=device,
            max_len=int(manifest.get("sparse_max_len", 192)),
            top_n=int(manifest.get("sparse_top_n", 128)),
        )[0]
        sparse_hits_by_qid[q["query_id"]] = sparse_search(sparse_matrix, idx, val, args.top_k_sparse)
    del sp_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    out = []
    stats_unique, stats_overlap = [], []
    for q in queries:
        qt = q["query_type"]
        d_hits = dense_hits_by_qid[q["query_id"]]
        s_hits = sparse_hits_by_qid[q["query_id"]]
        merged = merge_hits(d_hits, s_hits, qt, args.prefusion, args.rrf_k, args.w_dense, args.w_sparse)
        results = []
        for rank, (i, ds, ss, pf, src) in enumerate(merged, 1):
            results.append({
                "rank": rank, "product_id": product_ids[i],
                "dense_score": ds, "sparse_score": ss,
                "prefusion_score": pf, "source": src,
            })
        out.append({
            "query_id": q["query_id"], "query_type": qt,
            "run_name": "hybrid_dense_sparse_prefusion",
            "sparse_applicable": qt != "image", "results": results,
        })
        stats_unique.append(len(results))
        stats_overlap.append(len(set(i for i, _ in d_hits) & set(i for i, _ in s_hits)))
    write_jsonl(args.out, out)

    counts = {t: sum(q["query_type"] == t for q in queries) for t in ("text", "image", "image_text")}
    examples = []
    for rec in out[:3]:
        examples.append(f"- **{rec['query_id']}**: " + ", ".join(
            f"{h['product_id']} ({h['prefusion_score']:.3f})" for h in rec["results"][:3]))
    summary = [
        "# Stage 2 - Retrieval Summary", "",
        f"Queries by type: {counts}.",
        f"Dense top-K: {args.top_k_dense}; sparse top-K: {args.top_k_sparse}.",
        f"Prefusion method: {args.prefusion}; weighted fusion uses dense={args.w_dense}, sparse={args.w_sparse}; RRF k={args.rrf_k}.",
        f"Average unique candidates per query: {float(np.mean(stats_unique)):.2f}.",
        f"Average dense/sparse overlap per query: {float(np.mean(stats_overlap)):.2f}.",
        "", "## Example top results", "", *examples,
    ]
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
