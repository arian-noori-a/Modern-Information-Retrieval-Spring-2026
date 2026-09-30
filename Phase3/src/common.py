from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | os.PathLike[str], records: Iterable[dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def resolve_path(path: str | None, root: str | os.PathLike[str] = ".") -> str | None:
    if not path:
        return None
    p = Path(path)
    if p.is_absolute() and p.exists():
        return str(p)
    candidate = Path(root) / p
    if candidate.exists():
        return str(candidate.resolve())
    return str(candidate)


def minmax_map(hits: list[tuple[int, float]]) -> dict[int, float]:
    if not hits:
        return {}
    vals = np.asarray([s for _, s in hits], dtype=np.float32)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {i: 1.0 for i, _ in hits}
    return {i: float((s - lo) / (hi - lo)) for i, s in hits}


def zscore(values: list[float | None]) -> list[float]:
    arr = np.asarray([0.0 if v is None else float(v) for v in values], dtype=np.float32)
    if len(arr) == 0:
        return []
    sd = float(arr.std())
    if sd < 1e-9:
        return [0.0] * len(arr)
    mu = float(arr.mean())
    return [float((v - mu) / sd) for v in arr]


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def product_rerank_text(row: Any, max_chars: int = 3000) -> str:
    fields = [
        ("Title", getattr(row, "title", "")),
        ("Category", getattr(row, "category_path", "")),
        ("Product type", getattr(row, "product_type", "")),
        ("Brand", getattr(row, "brand", "")),
        ("Color", getattr(row, "color", "")),
        ("Material", getattr(row, "material", "")),
        ("Style", getattr(row, "style", "")),
        ("Features", getattr(row, "bullet_points", "")),
        ("Description", getattr(row, "description", "")),
    ]
    text = "\n".join(f"{k}: {safe_text(v)}" for k, v in fields if safe_text(v))
    return text[:max_chars]
