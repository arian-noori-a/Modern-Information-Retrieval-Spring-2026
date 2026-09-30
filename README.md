# Modern Information Retrieval — Spring 2026

Two practical project phases for **Modern Information Retrieval**, covering both classical information retrieval and neural/embedding-based methods.

## Phases

| Phase | Topics |
|---|---|
| **Phase 1** | Text preprocessing, inverted indexing, document-length/metadata/tiered indexes, vector-space scoring, BM25, unigram language models, spell correction, snippets, MinHash/LSH, retrieval evaluation, and a Streamlit search interface |
| **Phase 2** | Word2Vec from scratch, FastText, book-genre classification, clustering, BERT full fine-tuning, LoRA, few-shot learning, and layer-wise embedding visualization |

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
```

## Phase 1

Phase 1 implements a classical search engine over book metadata. The project includes:

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

## Phase 2

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
