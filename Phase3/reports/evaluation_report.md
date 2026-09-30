# Stage 4 - Evaluation Report

This Stage 4 report evaluates all runs against the reviewed qrels pool.
Metrics include Precision, Recall, MAP, NDCG, and judged coverage.
Unjudged results are not silently presented as confirmed irrelevant; judged coverage is reported separately.

## Metrics

| run_name                   | query_type   |   num_queries |   precision_at_5 |   precision_at_10 |   recall_at_10 |   recall_at_20 |      map |   ndcg_at_5 |   ndcg_at_10 |   judged_coverage |
|:---------------------------|:-------------|--------------:|-----------------:|------------------:|---------------:|---------------:|---------:|------------:|-------------:|------------------:|
| dense_single_vector        | text         |            10 |         0.86     |          0.78     |       0.756749 |       0.850243 | 0.802032 |    0.876854 |     0.910535 |          0.62     |
| dense_single_vector        | image        |             5 |         0.92     |          0.82     |       1        |       1        | 0.913279 |    0.786526 |     0.89466  |          0.5      |
| dense_single_vector        | image_text   |             5 |         0.72     |          0.64     |       0.474444 |       0.564722 | 0.573963 |    0.607201 |     0.591944 |          0.66     |
| dense_single_vector        | all          |            20 |         0.84     |          0.755    |       0.746986 |       0.816302 | 0.772826 |    0.786859 |     0.826918 |          0.6      |
| sparse_splade              | text         |            10 |         0.7      |          0.61     |       0.565339 |       0.798489 | 0.633094 |    0.577034 |     0.633616 |          0.655    |
| sparse_splade              | image_text   |             5 |         0.48     |          0.48     |       0.411667 |       0.571389 | 0.460743 |    0.405973 |     0.492346 |          0.62     |
| sparse_splade              | all          |            15 |         0.626667 |          0.566667 |       0.514115 |       0.722789 | 0.575644 |    0.520014 |     0.586526 |          0.643333 |
| hybrid_prefusion           | text         |            10 |         0.88     |          0.78     |       0.743938 |       0.941058 | 0.825714 |    0.855854 |     0.891608 |          0.74     |
| hybrid_prefusion           | image        |             5 |         0.92     |          0.82     |       1        |       1        | 0.913279 |    0.786526 |     0.89466  |          0.5      |
| hybrid_prefusion           | image_text   |             5 |         0.4      |          0.48     |       0.321944 |       0.857222 | 0.528857 |    0.371313 |     0.430851 |          0.83     |
| hybrid_prefusion           | all          |            20 |         0.77     |          0.715    |       0.702455 |       0.934834 | 0.773391 |    0.717387 |     0.777182 |          0.7025   |
| hybrid_cross_encoder       | text         |            10 |         0.7      |          0.65     |       0.591781 |       0.883919 | 0.667196 |    0.596124 |     0.657828 |          0.68     |
| hybrid_cross_encoder       | image_text   |             5 |         0.48     |          0.48     |       0.499167 |       0.565833 | 0.418671 |    0.30906  |     0.461492 |          0.64     |
| hybrid_cross_encoder       | all          |            15 |         0.626667 |          0.593333 |       0.56091  |       0.777891 | 0.584354 |    0.500436 |     0.592383 |          0.666667 |
| hybrid_cross_encoder_fused | text         |            10 |         0.72     |          0.69     |       0.647019 |       0.918397 | 0.70875  |    0.634095 |     0.716441 |          0.7      |
| hybrid_cross_encoder_fused | image        |             5 |         0.92     |          0.82     |       1        |       1        | 0.913279 |    0.786526 |     0.89466  |          0.5      |
| hybrid_cross_encoder_fused | image_text   |             5 |         0.4      |          0.42     |       0.358333 |       0.617222 | 0.425239 |    0.317164 |     0.419843 |          0.67     |
| hybrid_cross_encoder_fused | all          |            20 |         0.69     |          0.655    |       0.663093 |       0.863504 | 0.689005 |    0.59297  |     0.686846 |          0.6425   |

Qrels judgments: 318 across 20 queries.

## Limitations

The qrels are pooled rather than exhaustive, so recall and MAP are estimates over the judged pool. Image-only queries have no sparse-only run, and the CPU-friendly text cross-encoder run excludes image-only queries; the fused run preserves multimodal dense visual evidence.
