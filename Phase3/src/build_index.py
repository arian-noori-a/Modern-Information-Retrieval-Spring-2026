"""Stage 1: build multimodal dense embeddings, SPLADE sparse vectors, and indexes.

The required dense policy is one vector produced directly from a combined
{"text": ..., "image": ...} input. Text and image vectors are never manually
averaged or concatenated.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import sparse as sp
# Uncomment/install dependencies in Colab or Kaggle as needed.
from tqdm.auto import tqdm # type: ignore
import torch # type: ignore
from sentence_transformers import SentenceTransformer # type: ignore
import faiss # type: ignore
from transformers import AutoModelForMaskedLM, AutoTokenizer # type: ignore

from common import resolve_path


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def load_dense_model(model_name: str, device: str):

    cfg = {"device": device, "trust_remote_code": True}
    if device.startswith("cuda"):
        try:
            model = SentenceTransformer(
                model_name,
                device=device,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.float16},
            )
            return model, "float16_cuda"
        except Exception:
            pass
    model = SentenceTransformer(model_name, **cfg)
    return model, "default_dtype"


def _load_rgb(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def encode_dense(model, inputs: list[object], *, batch_size: int, normalize: bool,
                 task: str = "document") -> np.ndarray:
    prepared = []
    for item in inputs:
        if isinstance(item, dict) and "image" in item:
            prepared.append({"text": item.get("text", ""), "image": _load_rgb(item["image"])})
        elif isinstance(item, str) and os.path.exists(item):
            prepared.append(_load_rgb(item))
        else:
            prepared.append(item)
    fn = getattr(model, "encode_document" if task == "document" else "encode_query", None)
    if fn is None:
        fn = model.encode
    arr = fn(
        prepared,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
    )
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    return l2_normalize(arr) if normalize else arr


def splade_batch(model, tokenizer, texts: list[str], *, device: str,
                 max_len: int, top_n: int) -> list[tuple[np.ndarray, np.ndarray]]:

    enc = tokenizer(
        texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        logits = model(**enc).logits
        weights = torch.log1p(torch.relu(logits))
        mask = enc["attention_mask"].unsqueeze(-1)
        weights = (weights * mask).amax(dim=1)
    out = []
    for row in weights:
        k = min(top_n, row.numel())
        vals, idx = torch.topk(row, k=k)
        keep = vals > 0
        idx = idx[keep].detach().cpu().numpy().astype(np.int32)
        vals = vals[keep].detach().cpu().numpy().astype(np.float32)
        order = np.argsort(-vals)
        out.append((idx[order], vals[order]))
    return out


def dense_topk(index, vec: np.ndarray, k: int) -> list[tuple[int, float]]:
    q = np.asarray(vec, dtype=np.float32).reshape(1, -1)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    scores, ids = index.search(np.ascontiguousarray(q), k)
    return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]


def sparse_topk(matrix: sp.csr_matrix, idx: np.ndarray, val: np.ndarray, k: int):
    q = sp.csr_matrix((val, (np.zeros(len(idx), dtype=int), idx)), shape=(1, matrix.shape[1]))
    scores = (matrix @ q.T).toarray().ravel()
    k = min(k, len(scores))
    order = np.argpartition(-scores, k - 1)[:k]
    order = order[np.argsort(-scores[order])]
    return [(int(i), float(scores[i])) for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", default="data/catalog_subset.parquet")
    ap.add_argument("--dense_model", default="Qwen/Qwen3-VL-Embedding-2B")
    ap.add_argument("--sparse_model", default="prithivida/Splade_PP_en_v2")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0")
    ap.add_argument("--dense_batch_size", type=int, default=1)
    ap.add_argument("--dense_chunk", type=int, default=16)
    ap.add_argument("--sparse_batch_size", type=int, default=2)
    ap.add_argument("--sparse_max_len", type=int, default=192)
    ap.add_argument("--sparse_top_n", type=int, default=128)
    ap.add_argument("--emb_dir", default="artifacts/embeddings")
    ap.add_argument("--index_dir", default="artifacts/index")
    ap.add_argument("--summary", default="reports/indexing_summary.md")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    catalog = pd.read_parquet(args.catalog).copy()
    root = Path(".").resolve()
    catalog["resolved_image"] = catalog["image_path"].map(lambda p: resolve_path(p, root))
    missing = [p for p in catalog["resolved_image"] if not p or not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} catalog images are missing. Extract data/AIR_Phase3_stage1_images.zip first.")

    emb_dir, index_dir = Path(args.emb_dir), Path(args.index_dir)
    emb_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    dense_path = emb_dir / "product_dense.npy"
    ids_path = emb_dir / "product_ids.txt"
    sparse_jsonl = emb_dir / "product_sparse.jsonl"
    dense_index_path = index_dir / "dense_index.faiss"
    sparse_index_path = index_dir / "sparse_index.npz"
    manifest_path = index_dir / "index_manifest.json"

    ids = catalog["product_id"].astype(str).tolist()
    inputs = [
        {"text": str(t), "image": p}
        for t, p in zip(catalog["product_text"].fillna(""), catalog["resolved_image"])
    ]

    dense_model, load_cfg = load_dense_model(args.dense_model, device)
    t0 = time.time()
    if dense_path.exists() and ids_path.exists():
        dense = np.load(dense_path)
        if ids_path.read_text(encoding="utf-8").split() != ids:
            raise ValueError("Cached product_ids.txt does not align with the catalog.")
    else:
        parts = []
        for s in tqdm(range(0, len(inputs), args.dense_chunk), desc="dense products"):
            part = encode_dense(
                dense_model, inputs[s:s + args.dense_chunk], batch_size=args.dense_batch_size,
                normalize=True, task="document"
            )
            parts.append(part)
        dense = l2_normalize(np.vstack(parts).astype(np.float32))
        np.save(dense_path, dense)
        ids_path.write_text("\n".join(ids) + "\n", encoding="utf-8")
    dense_minutes = (time.time() - t0) / 60.0
    dense_dim = int(dense.shape[1])

    dense_index = faiss.IndexFlatIP(dense_dim)
    dense_index.add(np.ascontiguousarray(dense, dtype=np.float32))
    faiss.write_index(dense_index, str(dense_index_path))

    # Dense sanity checks while the dense model is still in memory.
    sanity = []
    for kind, mode in [("DENSE image-query", "image"), ("DENSE text-query", "text")]:
        chosen = None
        for r in range(min(len(catalog), 100)):
            inp = catalog.iloc[r]["resolved_image"] if mode == "image" else str(catalog.iloc[r]["product_text"])
            qv = encode_dense(dense_model, [inp], batch_size=1, normalize=True, task="query")[0]
            hits = dense_topk(dense_index, qv, 5)
            rank = next((k for k, (i, _) in enumerate(hits) if i == r), None)
            if rank is not None:
                chosen = (r, rank, hits); break
        if chosen is None:
            r = 0
            inp = catalog.iloc[r]["resolved_image"] if mode == "image" else str(catalog.iloc[r]["product_text"])
            qv = encode_dense(dense_model, [inp], batch_size=1, normalize=True, task="query")[0]
            hits = dense_topk(dense_index, qv, 5)
            rank = next((k for k, (i, _) in enumerate(hits) if i == r), None)
        else:
            r, rank, hits = chosen
        sanity.append((kind, ids[r], rank, [ids[i] for i, _ in hits[:3]]))

    # Free the 2B dense model before loading SPLADE to lower peak RAM on CPU.
    import gc
    del dense_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(args.sparse_model)
    sparse_model = AutoModelForMaskedLM.from_pretrained(args.sparse_model).to(device).eval()
    vocab = int(sparse_model.config.vocab_size)
    rows, cols, data, nnz = [], [], [], []
    with open(sparse_jsonl, "w", encoding="utf-8") as fh:
        texts = catalog["product_text"].fillna("").astype(str).tolist()
        for s in tqdm(range(0, len(texts), args.sparse_batch_size), desc="sparse products"):
            batch = splade_batch(
                sparse_model, tokenizer, texts[s:s + args.sparse_batch_size],
                device=device, max_len=args.sparse_max_len, top_n=args.sparse_top_n,
            )
            for j, (idx, val) in enumerate(batch):
                r = s + j
                nnz.append(len(idx))
                rows.extend([r] * len(idx)); cols.extend(idx.tolist()); data.extend(val.tolist())
                fh.write(json.dumps({"product_id": ids[r], "indices": idx.tolist(), "values": val.tolist()}) + "\n")
    sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(ids), vocab), dtype=np.float32)
    sp.save_npz(sparse_index_path, sparse_matrix)

    # Put a successful sparse self-retrieval first so the summary is meaningful.
    sparse_probes = []
    chosen = None
    for r in range(min(len(catalog), 100)):
        idx, val = splade_batch(
            sparse_model, tokenizer, [str(catalog.iloc[r]["title"])], device=device,
            max_len=args.sparse_max_len, top_n=args.sparse_top_n,
        )[0]
        hits = sparse_topk(sparse_matrix, idx, val, 5)
        rank = next((k for k, (i, _) in enumerate(hits) if i == r), None)
        if rank is not None:
            chosen = (r, rank, hits); break
    if chosen is not None:
        sparse_probes.append(chosen)
    for r in [len(catalog) // 2, len(catalog) - 1]:
        idx, val = splade_batch(
            sparse_model, tokenizer, [str(catalog.iloc[r]["title"])], device=device,
            max_len=args.sparse_max_len, top_n=args.sparse_top_n,
        )[0]
        hits = sparse_topk(sparse_matrix, idx, val, 5)
        rank = next((k for k, (i, _) in enumerate(hits) if i == r), None)
        sparse_probes.append((r, rank, hits))
    for r, rank, hits in sparse_probes:
        sanity.append(("SPARSE title-query", ids[r], rank, [ids[i] for i, _ in hits[:3]]))

    manifest = {
        "num_products": int(len(catalog)),
        "dense_model": args.dense_model,
        "dense_input_format": "multimodal_dict_text_image",
        "dense_embedding_policy": "single_vector_from_combined_text_image_input",
        "manual_text_image_vector_mixing": False,
        "dense_encode_method": "encode_document_or_encode_fallback",
        "dense_load_config": load_cfg,
        "dense_batch_size": args.dense_batch_size,
        "sparse_model": args.sparse_model,
        "sparse_max_len": args.sparse_max_len,
        "dense_dimension": dense_dim,
        "sparse_top_n": args.sparse_top_n,
        "sparse_vocab_size": vocab,
        "vector_store": "faiss_plus_csr",
        "similarity": "FAISS IndexFlatIP over L2-normalized vectors; sparse dot product",
        "id_mapping_file": str(ids_path),
        "runtime": f"{platform.system()} {platform.machine()} / {device}",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "# Stage 1 - Indexing Summary", "",
        f"Dense model: `{args.dense_model}`; load configuration: `{load_cfg}`.",
        f"Sparse model: `{args.sparse_model}`.",
        f"Dense dimension: {dense_dim}; dense batch size: {args.dense_batch_size}; runtime environment: {manifest['runtime']}.",
        f"Product vectors: {len(ids)}; dense runtime: {dense_minutes:.2f} minutes.",
        f"Sparse top-N: {args.sparse_top_n}; average non-zero dimensions: {float(np.mean(nnz)):.2f}.",
        "Index type: FAISS IndexFlatIP; similarity: inner product on L2-normalized vectors (cosine-equivalent). Sparse index: SciPy CSR with dot-product search.",
        "", "## Sanity-check searches", "",
    ]
    for kind, pid, rank, top in sanity:
        lines.append(f"- {kind}: product={pid}; self rank={rank if rank is not None else 'None'}; top={top}")
    Path(args.summary).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Built Stage 1 artifacts for {len(ids)} products on {device}.")


if __name__ == "__main__":
    main()
