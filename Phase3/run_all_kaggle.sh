#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1

# Full Kaggle GPU runner for AIR Phase 3, Stages 0 through 5.
# Run from the writable project root, e.g. /kaggle/working/AIR_Phase3_completed.
# Optional overrides:
#   DENSE_BATCH_SIZE=1 SPARSE_BATCH_SIZE=8 RERANK_BATCH_SIZE=16 bash run_all_kaggle.sh
#   LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct LLM_TOP_K=8 bash run_all_kaggle.sh

DENSE_BATCH_SIZE="${DENSE_BATCH_SIZE:-2}"
DENSE_CHUNK="${DENSE_CHUNK:-32}"
SPARSE_BATCH_SIZE="${SPARSE_BATCH_SIZE:-16}"
RERANK_BATCH_SIZE="${RERANK_BATCH_SIZE:-32}"
LLM_MODEL="${LLM_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
LLM_TOP_K="${LLM_TOP_K:-8}"
LLM_MAX_NEW_TOKENS="${LLM_MAX_NEW_TOKENS:-700}"

python - <<'PY'
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('ERROR: CUDA is not available. Enable a Kaggle GPU accelerator before running.')
print('GPU:', torch.cuda.get_device_name(0))
print('CUDA runtime:', torch.version.cuda)
PY

echo "============================================================"
echo "[0/5] Stage 0 - data acceptance tests"
echo "============================================================"
python src/tests.py --stage 0 --catalog data/catalog_subset.parquet

echo "============================================================"
echo "[1/5] Stage 1 - dense multimodal + SPLADE indexes"
echo "============================================================"
python src/build_index.py \
  --device cuda \
  --dense_batch_size "$DENSE_BATCH_SIZE" \
  --dense_chunk "$DENSE_CHUNK" \
  --sparse_batch_size "$SPARSE_BATCH_SIZE" \
  --sparse_max_len 192 \
  --sparse_top_n 128
python src/tests.py --stage 1 \
  --catalog data/catalog_subset.parquet \
  --emb_dir artifacts/embeddings \
  --index_dir artifacts/index \
  --summary reports/indexing_summary.md

echo "============================================================"
echo "[2/5] Stage 2 - hybrid multimodal retrieval"
echo "============================================================"
python src/retrieve.py \
  --device cuda \
  --top_k_dense 100 \
  --top_k_sparse 100 \
  --out outputs/retrieval_results.jsonl
python src/tests.py --stage 2

echo "============================================================"
echo "[3/5] Stage 3 - cross-encoder reranking + dense evidence fusion"
echo "============================================================"
python src/rerank.py \
  --device cuda \
  --reranker cross-encoder/ms-marco-MiniLM-L-6-v2 \
  --rerank_top_n 100 \
  --batch_size "$RERANK_BATCH_SIZE" \
  --out outputs/reranked_results.jsonl
python src/tests.py --stage 3

echo "============================================================"
echo "[4/5] Stage 4 - evaluation with reviewed qrels"
echo "============================================================"
python src/evaluate.py \
  --queries data/queries.jsonl \
  --qrels data/qrels.jsonl \
  --retrieval outputs/retrieval_results.jsonl \
  --reranked outputs/reranked_results.jsonl \
  --out reports/evaluation_metrics.csv \
  --report reports/evaluation_report.md \
  --error_analysis reports/error_analysis.md
python src/tests.py --stage 4

echo "============================================================"
echo "[5/5] Stage 5 - local LLM structured JSON generation"
echo "============================================================"
python src/generate_answer.py \
  --device cuda \
  --model "$LLM_MODEL" \
  --top_k "$LLM_TOP_K" \
  --max_new_tokens "$LLM_MAX_NEW_TOKENS" \
  --repair_attempts 2 \
  --out outputs/final_answers.jsonl \
  --summary reports/llm_output_summary.md
python src/tests.py --stage 5 --top_k "$LLM_TOP_K"

echo "============================================================"
echo "ALL STAGES 0-5 COMPLETED SUCCESSFULLY"
echo "============================================================"
python - <<'PY'
import os, shutil
root = os.getcwd()
out = '/kaggle/working/AIR_Phase3_FULL_RESULTS'
if os.path.exists('/kaggle/working'):
    shutil.make_archive(out, 'zip', root)
    print('Packaged complete project at:')
    print(out + '.zip')
else:
    print('Not running under Kaggle; skipping automatic ZIP packaging.')
PY
