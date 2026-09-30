# Stage 1 - Indexing Summary

Dense model: `Qwen/Qwen3-VL-Embedding-2B`; load configuration: `float16_cuda`.
Sparse model: `prithivida/Splade_PP_en_v2`.
Dense dimension: 2048; dense batch size: 1; runtime environment: Linux x86_64 / cuda.
Product vectors: 2000; dense runtime: 0.00 minutes.
Sparse top-N: 128; average non-zero dimensions: 120.41.
Index type: FAISS IndexFlatIP; similarity: inner product on L2-normalized vectors (cosine-equivalent). Sparse index: SciPy CSR with dot-product search.

## Sanity-check searches

- DENSE image-query: product=B07B51H7B1|amazon.com; self rank=1; top=['B086TGRSBG|amazon.com', 'B07B51H7B1|amazon.com', 'B07B4W2MFW|amazon.com']
- DENSE text-query: product=B07B51H7B1|amazon.com; self rank=0; top=['B07B51H7B1|amazon.com', 'B075X2CVQK|amazon.com', 'B075X2LMTC|amazon.com']
- SPARSE title-query: product=B07B51H7B1|amazon.com; self rank=0; top=['B07B51H7B1|amazon.com', 'B075X2CVQK|amazon.com', 'B0719SNKSY|amazon.com']
- SPARSE title-query: product=B07SYL8MPN|amazon.sa; self rank=0; top=['B07SYL8MPN|amazon.sa', 'B07SYL958P|amazon.sg', 'B07T25P4BB|amazon.com']
- SPARSE title-query: product=B075X25BYC|amazon.com; self rank=0; top=['B075X25BYC|amazon.com', 'B07QX2BYDH|amazon.com', 'B08DWMY83P|amazon.com']
