"""Create the required 20-query starter set and non-identical transformed query images."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance, ImageOps


def main():
    root = Path(".")
    df = pd.read_parquet(root / "data/catalog_subset.parquet")
    out_dir = root / "data/query_images"; out_dir.mkdir(parents=True, exist_ok=True)
    refs = {}
    for ptype in ["CHAIR", "LAMP", "SOFA", "CABINET", "TABLE"]:
        row = df[df["product_type"] == ptype].iloc[0]
        src = root / row["image_path"]
        img = Image.open(src).convert("RGB")
        # Make a visibly similar but non-identical query image: mirror + small border crop + contrast change.
        img = ImageOps.mirror(img)
        w, h = img.size
        dx, dy = max(1, w // 25), max(1, h // 25)
        img = img.crop((dx, dy, w - dx, h - dy)).resize((w, h))
        img = ImageEnhance.Contrast(img).enhance(0.92)
        dest = out_dir / f"query_{ptype.lower()}.jpg"
        img.save(dest, quality=90)
        refs[ptype] = str(dest).replace("\\", "/")

    queries = [
        {"query_id":"q001","query_text":"gray or beige fabric sofa for a small living room, not leather","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q002","query_text":"modern wooden storage cabinet or sideboard, light color","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q003","query_text":"black office chair with armrests, not a dining chair","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q004","query_text":"white or light wood shelf for bedroom storage","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q005","query_text":"floor lamp for a living room, not a table lamp","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q006","query_text":"round dining table suitable for four people","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q007","query_text":"wooden queen bed frame with headboard, not metal","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q008","query_text":"compact writing desk for a home office with storage","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q009","query_text":"small ottoman for living room seating or footrest, neutral color","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q010","query_text":"bedroom dresser with multiple drawers, not an open shelf","query_image_path":None,"query_type":"text","split":"test"},
        {"query_id":"q011","query_text":None,"query_image_path":refs["CHAIR"],"query_type":"image","split":"test"},
        {"query_id":"q012","query_text":None,"query_image_path":refs["LAMP"],"query_type":"image","split":"test"},
        {"query_id":"q013","query_text":None,"query_image_path":refs["SOFA"],"query_type":"image","split":"test"},
        {"query_id":"q014","query_text":None,"query_image_path":refs["CABINET"],"query_type":"image","split":"test"},
        {"query_id":"q015","query_text":None,"query_image_path":refs["TABLE"],"query_type":"image","split":"test"},
        {"query_id":"q016","query_text":"similar chair style, but black and not intended as a dining chair","query_image_path":refs["CHAIR"],"query_type":"image_text","split":"test"},
        {"query_id":"q017","query_text":"similar lamp shape, but suitable as a floor lamp, not a table lamp","query_image_path":refs["LAMP"],"query_type":"image_text","split":"test"},
        {"query_id":"q018","query_text":"similar sofa silhouette, but fabric and not leather","query_image_path":refs["SOFA"],"query_type":"image_text","split":"test"},
        {"query_id":"q019","query_text":"similar storage cabinet, but in white or light wood","query_image_path":refs["CABINET"],"query_type":"image_text","split":"test"},
        {"query_id":"q020","query_text":"similar table shape, but a compact wooden dining table","query_image_path":refs["TABLE"],"query_type":"image_text","split":"test"},
    ]
    with open(root / "data/queries.jsonl", "w", encoding="utf-8") as fh:
        for q in queries: fh.write(json.dumps(q) + "\n")
    print("Wrote 20 queries and 5 transformed query images.")


if __name__ == "__main__":
    main()
