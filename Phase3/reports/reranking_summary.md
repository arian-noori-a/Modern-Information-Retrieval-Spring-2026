# Stage 3 - Reranking Summary

Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
Reranker input format: `(query_text, compact product metadata text)` for text-bearing queries.
Image-only queries are not sent to the text cross-encoder; their final ordering retains the multimodal dense visual score.
Reranking depth: 100.
Score normalization method: per-query z-score.
Fusion weights: {'text': {'ce': 0.65, 'dense': 0.2, 'sparse': 0.15}, 'image': {'ce': 0.0, 'dense': 1.0, 'sparse': 0.0}, 'image_text': {'ce': 0.55, 'dense': 0.3, 'sparse': 0.15}}.
Runtime environment/device: cuda.

This CPU-friendly main fused run preserves visual evidence for image and image+text queries through the Stage 1/2 multimodal dense score.

## Before/after examples

- **q001** (text): before=['B086B5MRFB|amazon.in', 'B07HPNBTHF|amazon.co.uk', 'B07B4MRLT8|amazon.com']; after=['B07HPNBTHF|amazon.co.uk', 'B082Q8NLZ3|amazon.com', 'B07B4MRLT8|amazon.com']
- **q002** (text): before=['B07H8SSY4F|amazon.co.uk', 'B07B813LW1|amazon.com', 'B082PFXVPB|amazon.com']; after=['B07H8SSY4F|amazon.co.uk', 'B07B77VXN8|amazon.com', 'B07B813LW1|amazon.com']
- **q003** (text): before=['B07SSHYD2Z|amazon.co.uk', 'B081HWKWN6|amazon.co.uk', 'B081HX9SRM|amazon.co.uk']; after=['B07SSHYD2Z|amazon.co.uk', 'B081HX9SRM|amazon.co.uk', 'B081HWKWN6|amazon.co.uk']
- **q016** (image_text): before=['B07MBFDQLY|amazon.com', 'B084W339RK|amazon.com', 'B07HSBJ5D6|amazon.com']; after=['B07B83KVGK|amazon.com', 'B084S1DM72|amazon.com', 'B084W339RK|amazon.com']
