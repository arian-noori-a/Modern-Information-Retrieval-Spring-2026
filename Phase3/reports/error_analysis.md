# Stage 4 - Error Analysis

## Dense retrieval works better than sparse retrieval

Query: gray or beige fabric sofa for a small living room, not leather
Query type: text
Top results before reranking: B086B5MRFB|amazon.in (rel=2), B07HPNBTHF|amazon.co.uk (rel=0), B07B4MRLT8|amazon.com (rel=1), B07QJXW4JR|amazon.com (rel=1), B073G8Q656|amazon.com (rel=2)
Top results after reranking: B07HPNBTHF|amazon.co.uk (rel=0), B082Q8NLZ3|amazon.com (rel=1), B07B4MRLT8|amazon.com (rel=1), B086B5MRFB|amazon.in (rel=2), B07QJXW4JR|amazon.com (rel=1)
Explanation: compare the graded relevance labels above with the change in ordering. Dense evidence tends to help semantic/visual similarity, sparse evidence helps exact lexical attributes, and the cross-encoder can improve fine-grained text matching but may also over-weight textual cues. This case should be reviewed alongside the product image for image-bearing queries.

## Sparse retrieval works better than dense retrieval

Query: modern wooden storage cabinet or sideboard, light color
Query type: text
Top results before reranking: B07H8SSY4F|amazon.co.uk (rel=2), B07B813LW1|amazon.com (rel=1), B082PFXVPB|amazon.com (rel=1), B07QFB1L2P|amazon.com (rel=1), B07B77VXN8|amazon.com (rel=1)
Top results after reranking: B07H8SSY4F|amazon.co.uk (rel=2), B07B77VXN8|amazon.com (rel=1), B07B813LW1|amazon.com (rel=1), B082PFXVPB|amazon.com (rel=1), B07QFB1L2P|amazon.com (rel=1)
Explanation: compare the graded relevance labels above with the change in ordering. Dense evidence tends to help semantic/visual similarity, sparse evidence helps exact lexical attributes, and the cross-encoder can improve fine-grained text matching but may also over-weight textual cues. This case should be reviewed alongside the product image for image-bearing queries.

## Cross-encoder reranking improves ranking

Query: gray or beige fabric sofa for a small living room, not leather
Query type: text
Top results before reranking: B086B5MRFB|amazon.in (rel=2), B07HPNBTHF|amazon.co.uk (rel=0), B07B4MRLT8|amazon.com (rel=1), B07QJXW4JR|amazon.com (rel=1), B073G8Q656|amazon.com (rel=2)
Top results after reranking: B07HPNBTHF|amazon.co.uk (rel=0), B082Q8NLZ3|amazon.com (rel=1), B07B4MRLT8|amazon.com (rel=1), B086B5MRFB|amazon.in (rel=2), B07QJXW4JR|amazon.com (rel=1)
Explanation: compare the graded relevance labels above with the change in ordering. Dense evidence tends to help semantic/visual similarity, sparse evidence helps exact lexical attributes, and the cross-encoder can improve fine-grained text matching but may also over-weight textual cues. This case should be reviewed alongside the product image for image-bearing queries.

## Cross-encoder reranking hurts or changes ranking

Query: black office chair with armrests, not a dining chair
Query type: text
Top results before reranking: B07SSHYD2Z|amazon.co.uk (rel=2), B081HWKWN6|amazon.co.uk (rel=2), B081HX9SRM|amazon.co.uk (rel=2), B07QLGRVPN|amazon.in (rel=1), B075Z6S94K|amazon.com (rel=1)
Top results after reranking: B07SSHYD2Z|amazon.co.uk (rel=2), B081HX9SRM|amazon.co.uk (rel=2), B081HWKWN6|amazon.co.uk (rel=2), B07QLGRVPN|amazon.in (rel=1), B07R41343J|amazon.co.uk (rel=0)
Explanation: compare the graded relevance labels above with the change in ordering. Dense evidence tends to help semantic/visual similarity, sparse evidence helps exact lexical attributes, and the cross-encoder can improve fine-grained text matching but may also over-weight textual cues. This case should be reviewed alongside the product image for image-bearing queries.

## Image or image+text query case

Query: data/query_images/query_chair.jpg
Query type: image
Top results before reranking: B07HYYJKMD|amazon.com (rel=1), B082VMWB4G|amazon.com (rel=1), B075X5TN2K|amazon.com (rel=1), B0792JGRNS|amazon.ca (rel=0), B07B4GXYDG|amazon.com (rel=1)
Top results after reranking: B07HYYJKMD|amazon.com (rel=1), B082VMWB4G|amazon.com (rel=1), B075X5TN2K|amazon.com (rel=1), B0792JGRNS|amazon.ca (rel=0), B07B4GXYDG|amazon.com (rel=1)
Explanation: compare the graded relevance labels above with the change in ordering. Dense evidence tends to help semantic/visual similarity, sparse evidence helps exact lexical attributes, and the cross-encoder can improve fine-grained text matching but may also over-weight textual cues. This case should be reviewed alongside the product image for image-bearing queries.
