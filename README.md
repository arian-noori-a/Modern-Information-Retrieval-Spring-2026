# Modern Information Retrieval — Spring 2026

Practical project phases for **Modern Information Retrieval**, covering classical retrieval, neural representation learning, and multimodal retrieval with reranking and answer generation.

## Phases

| Phase | Topics |
|---|---|
| **Phase 1** | Text preprocessing, inverted indexing, document-length/metadata/tiered indexes, vector-space scoring, BM25, unigram language models, spell correction, snippets, MinHash/LSH, retrieval evaluation, and a Streamlit search interface |
| **Phase 2** | Word2Vec from scratch, FastText, book-genre classification, clustering, BERT full fine-tuning, LoRA, few-shot learning, and layer-wise embedding visualization |
| **Phase 3** | Multimodal product retrieval with Qwen3-VL dense embeddings and SPLADE, hybrid fusion, cross-encoder reranking, pooled-qrels evaluation, and structured LLM answer generation |

## Repository Structure

```text
Phase1/
  build.py
  requirements.txt
  stopwords.txt
  top_3000_rated_books.csv
  Logic/
  UI/

Phase2/
  word_embeddings.ipynb
  genre_classification_and_clustering.ipynb
  bert_finetuning.ipynb
  layerwise_bert_visualizations/

Phase3/
  phase3_runner.ipynb
  phase3_analysis.ipynb
  check_kaggle_gpu.py
  requirements_kaggle.txt
  run_all_kaggle.sh
  src/
  reports/
```

## Phase 1 — Classical Retrieval

Phase 1 implements a classical search engine over book metadata, including:

- preprocessing and stemming
- inverted indexes for descriptions, genres, and characters
- document-length, metadata, and tiered indexes
- SMART-style vector-space scoring
- Okapi BM25
- unigram language-model scoring with smoothing
- k-gram/Jaccard spell correction
- query-focused snippets
- MinHash/LSH near-duplicate detection
- Precision, Recall, F1, AP/MAP, DCG/NDCG, and RR/MRR evaluation
- a Streamlit search UI

Generated files such as `crawled.json`, `preprocessed.json`, and the `index/` directory are intentionally excluded and can be rebuilt from the included source dataset with `build.py`.

## Phase 2 — Neural Representations

Phase 2 explores representation learning and neural retrieval-related methods:

- Word2Vec implemented from scratch with skip-gram and negative sampling
- FastText with Gensim and the native FastText package
- genre classification using TF-IDF and sentence embeddings
- scratch KNN, softmax logistic regression, and MLP classifiers
- K-Means, spherical K-Means, DBSCAN, and hierarchical agglomerative clustering
- BERT full fine-tuning
- parameter-efficient LoRA fine-tuning
- few-shot classification with a frozen BERT encoder
- layer-wise PCA and silhouette analysis of BERT representations

Selected generated BERT visualization figures are included. Large/generated intermediate datasets and model artifacts are excluded.

## Phase 3 — Multimodal Retrieval and Answer Generation

Phase 3 implements a staged multimodal product-search pipeline:

1. **Multimodal indexing** — a Qwen3-VL embedding model produces one dense vector from combined product text and image input, while SPLADE produces sparse lexical representations.
2. **Hybrid retrieval** — FAISS dense search and sparse retrieval are fused for text, image, and image+text queries.
3. **Reranking** — a cross-encoder reranks text-bearing candidates and combines its score with dense/sparse evidence; image-only queries retain the multimodal dense ranking.
4. **Evaluation** — pooled relevance judgments are evaluated with Precision, Recall, MAP, NDCG, and judged coverage, with qualitative error analysis.
5. **Answer generation** — the top reranked products are passed to a local instruction-following LLM for structured JSON recommendations with validation and deterministic fallback.

The supplied evaluation uses 20 text/image/image+text queries and a reviewed pool of 318 relevance judgments. Compact reports are included, while the underlying datasets, embeddings, indexes, raw retrieval outputs, and model artifacts are intentionally excluded.
