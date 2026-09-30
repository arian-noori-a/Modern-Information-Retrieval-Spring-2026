# Stage 0 - Data Preparation Summary

The supplied course subset contains 2,000 products in `data/catalog_subset.parquet`. Each row has a unique `product_id`, non-empty title and `product_text`, a local small-image path, and product metadata such as product type, category, brand, color, material, style, dimensions, description, and bullet points where available.

The image archive `data/AIR_Phase3_stage1_images.zip` is extracted into `data/raw/images/small/...`. The project uses these small images rather than the full ABO image archive. The fixed catalog and the supplied `prep_stats.json` are treated as Stage 0 inputs; no resampling is performed in later stages.
