# Stage 5 - LLM Output Summary

## Configuration

- LLM model: `Qwen/Qwen2.5-1.5B-Instruct`
- Device: `cuda`
- Top reranked products per query: 8
- Maximum repair attempts: 2
- Prompting template: JSON-only product-search assistant prompt containing the query, image-presence flag, top product metadata, allowed role labels, allowed decision labels, schema instructions, negative-constraint guidance, and a no-invention rule.

## Results

- Number of generated answers: 20
- Raw JSON validity rate: 0.0%
- Schema-valid LLM rate before deterministic fallback: 0.0%
- Repaired outputs: 0
- Deterministic fallback outputs: 20
- Final JSON validity after repair/fallback: 100.0%
- Final schema-valid rate after repair/fallback: 100.0%

## Example Final Answers

```json
{
  "query_id": "q001",
  "interpreted_need": {
    "category": "sofa",
    "use_case": "gray or beige fabric sofa for a small living room, not leather",
    "positive_preferences": [
      "gray or beige fabric sofa",
      "a small living room",
      "not leather"
    ],
    "negative_constraints": [
      "not leather"
    ],
    "visual_preferences": [],
    "uncertain_fields": [
      "attributes not present in the catalog metadata"
    ]
  },
  "product_judgements": [
    {
      "product_id": "B07HPNBTHF|amazon.co.uk",
      "role": "exact",
      "evidence": [
        "Amazon Brand - Alkove Porto Hand-Crafted Leather Sofa, 3 Seater, 97 x 210 x 90cm, Mineral",
        "SOFA",
        "Beige"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    },
    {
      "product_id": "B082Q8NLZ3|amazon.com",
      "role": "substitute",
      "evidence": [
        "Amazon Brand – Stone & Beam Conner Slipcover Sofa Couch, 90\"W, Beige",
        "FURNITURE_COVER",
        "Beige"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    },
    {
      "product_id": "B07B4MRLT8|amazon.com",
      "role": "substitute",
      "evidence": [
        "Amazon Brand – Stone & Beam Bartow Living Room Sectional Sofa Couch with Slipcover, 95.5\"W, Grey",
        "FURNITURE_COVER",
        "Grey"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    }
  ],
  "decision": "recommend_exact_with_warning",
  "customer_response": "The strongest available candidate is Amazon Brand - Alkove Porto Hand-Crafted Leather Sofa, 3 Seater, 97 x 210 x 90cm, Mineral. It best matches the provided search evidence, while unspecified attributes should be treated as uncertain."
}
```

```json
{
  "query_id": "q002",
  "interpreted_need": {
    "category": "cabinet",
    "use_case": "modern wooden storage cabinet or sideboard, light color",
    "positive_preferences": [
      "modern wooden storage cabinet or sideboard",
      "light color"
    ],
    "negative_constraints": [],
    "visual_preferences": [],
    "uncertain_fields": [
      "attributes not present in the catalog metadata"
    ]
  },
  "product_judgements": [
    {
      "product_id": "B07H8SSY4F|amazon.co.uk",
      "role": "exact",
      "evidence": [
        "Amazon Brand - Movian Moselle 3-Drawer 2-Door Sideboard Storage Cabinet, 165 x 78 x 50cm, Light Brown Oak-Effect/White",
        "CABINET",
        "Oak Sonoma/Alpine White"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    },
    {
      "product_id": "B07B77VXN8|amazon.com",
      "role": "substitute",
      "evidence": [
        "Amazon Brand – Rivet Modern Wood Buffet Bar Cabinet Credenza with Gold Accents, 35 Inch Height, Brown",
        "CABINET",
        "Brown"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    },
    {
      "product_id": "B07B813LW1|amazon.com",
      "role": "substitute",
      "evidence": [
        "Amazon Brand – Rivet Mid-Century Modern Industrial Metal Buffet Credenza, 28 Inch Height, Beige, Wood",
        "CABINET",
        "Beige"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    }
  ],
  "decision": "recommend_exact_with_warning",
  "customer_response": "The strongest available candidate is Amazon Brand - Movian Moselle 3-Drawer 2-Door Sideboard Storage Cabinet, 165 x 78 x 50cm, Light Brown Oak-Effect/White. It best matches the provided search evidence, while unspecified attributes should be treated as uncertain."
}
```

```json
{
  "query_id": "q003",
  "interpreted_need": {
    "category": "home furniture and decor",
    "use_case": "black office chair with armrests, not a dining chair",
    "positive_preferences": [
      "black office chair",
      "armrests",
      "not a dining chair"
    ],
    "negative_constraints": [
      "not a dining chair"
    ],
    "visual_preferences": [],
    "uncertain_fields": [
      "attributes not present in the catalog metadata"
    ]
  },
  "product_judgements": [
    {
      "product_id": "B07SSHYD2Z|amazon.co.uk",
      "role": "exact",
      "evidence": [
        "AmazonBasics Swivel Office Chair with Medium Back and Armrest, Mesh Back, Black",
        "HOME_FURNITURE_AND_DECOR"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    },
    {
      "product_id": "B081HX9SRM|amazon.co.uk",
      "role": "substitute",
      "evidence": [
        "UMI. by Amazon - Ergonomic office chair, Breathable High-Back Mesh Desk Chair with Adjustable Lumbar Support, 3D Armrests, Tilt Function and 360° Rotating Wheels",
        "ACCESSORY_OR_PART_OR_SUPPLY",
        "Black"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    },
    {
      "product_id": "B081HWKWN6|amazon.co.uk",
      "role": "substitute",
      "evidence": [
        "UMI. by Amazon - Ergonomic office chair, Swivelling Mesh Computer Chair with Adjustable Lumbar Support and PU Armrests, Breathable Mesh Back and Padded Seat, 150 kg Max Load",
        "ACCESSORY_OR_PART_OR_SUPPLY",
        "Black"
      ],
      "constraint_violations": [],
      "reason": "This item is among the highest reranked candidates and matches available catalog evidence; unknown attributes are not assumed."
    }
  ],
  "decision": "recommend_exact_with_warning",
  "customer_response": "The strongest available candidate is AmazonBasics Swivel Office Chair with Medium Back and Armrest, Mesh Back, Black. It best matches the provided search evidence, while unspecified attributes should be treated as uncertain."
}
```

## Failure Case / Limitations

The LLM receives product metadata rather than raw product images at Stage 5, so it must not invent visual details. Image evidence is preserved by the upstream multimodal retrieval and fused reranking stages, while Stage 5 only structures/explains those candidates. Small local LLMs may occasionally emit malformed JSON or unsupported claims; the validator, repair loop, and deterministic fallback prevent such outputs from entering the final artifact.
