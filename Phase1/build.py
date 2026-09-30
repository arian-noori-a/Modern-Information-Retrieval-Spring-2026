import json
import shutil
from pathlib import Path

from Logic.preprocess import csv_to_json, preprocess_docs
from Logic.spell_correction import SpellCorrection
from Logic.indexer.index import Index
from Logic.indexer.indexes_enum import Indexes
from Logic.indexer.document_lengths_index import DocumentLengthsIndex
from Logic.indexer.metadata_index import Metadata_index
from Logic.indexer.tiered_index import Tiered_index

ROOT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT_DIR / "top_3000_rated_books.csv"
CRAWLED_JSON_PATH = ROOT_DIR / "crawled.json"
PREPROCESSED_JSON_PATH = ROOT_DIR / "preprocessed.json"
INDEX_DIR = ROOT_DIR / "index"
SPELL_MODEL_PATH = INDEX_DIR / "spell_correction.pkl"


def build_dataset_files():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {CSV_PATH}")

    print("[1/5] Creating crawled.json from CSV...")
    csv_to_json(CSV_PATH, CRAWLED_JSON_PATH)

    print("[2/5] Creating preprocessed.json...")
    with open(CRAWLED_JSON_PATH, "r", encoding="utf-8") as file:
        docs = json.load(file)
    preprocess_docs(docs)
    with open(PREPROCESSED_JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(docs, file, ensure_ascii=False, indent=2)

    return docs


def build_indexes(preprocessed_docs):
    print("[3/5] Building indexes...")
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index = Index(preprocessed_docs)
    index.store_index(str(INDEX_DIR) + "/", Indexes.DOCUMENTS.value)
    index.store_index(str(INDEX_DIR) + "/", Indexes.CHARACTERS.value)
    index.store_index(str(INDEX_DIR) + "/", Indexes.GENRES.value)
    index.store_index(str(INDEX_DIR) + "/", Indexes.DESCRIPTIONS.value)

    print("[4/5] Building document-length, metadata, and tiered indexes...")
    DocumentLengthsIndex(str(INDEX_DIR) + "/")
    Metadata_index(str(INDEX_DIR) + "/")
    Tiered_index(str(INDEX_DIR) + "/")


def build_spell_model():
    print("[5/5] Building spell-correction model...")
    with open(CRAWLED_JSON_PATH, "r", encoding="utf-8") as file:
        raw_docs = json.load(file)

    corpus = [
        (book.get("description") if isinstance(book.get("description"), str) else "") + " " +
        " ".join(book.get("genres", [])) + " " +
        " ".join(book.get("characters", []))
        for book in raw_docs
    ]
    SpellCorrection(all_documents=corpus, save_path=str(SPELL_MODEL_PATH))


def main():
    preprocessed_docs = build_dataset_files()
    build_indexes(preprocessed_docs)
    build_spell_model()
    print("\nDone. Generated files:")
    print(f"- {CRAWLED_JSON_PATH.name}")
    print(f"- {PREPROCESSED_JSON_PATH.name}")
    print(f"- {INDEX_DIR.name}/")
    print("\nYou can now run:")
    print("streamlit run UI/main.py")


if __name__ == "__main__":
    main()
