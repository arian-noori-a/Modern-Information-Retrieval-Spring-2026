from pathlib import Path
from typing import Dict, List
import json

from Logic.Search import SearchEngine
from Logic.indexer.indexes_enum import Indexes
from Logic.spell_correction import SpellCorrection

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "crawled.json"
INDEX_DIR = ROOT_DIR / "index"
SPELL_PKL_PATH = INDEX_DIR / "spell_correction.pkl"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

books_dataset = {str(book["id"]): book for book in data}

corpus = [
    (book.get("description") if isinstance(book.get("description"), str) else "") + " " +
    " ".join(book.get("genres", [])) + " " +
    " ".join(book.get("characters", []))
    for book in data
]

INDEX_DIR.mkdir(parents=True, exist_ok=True)
spell_correction_obj = SpellCorrection(
    all_documents=corpus,
    load_path=str(SPELL_PKL_PATH),
    save_path=str(SPELL_PKL_PATH),
)

search_engine = SearchEngine(path=str(INDEX_DIR) + "/")

def search(
    query: str,
    max_result_count: int,
    method: str = "ltn.lnn",
    weights: list = None,
    should_print=False,
    preferred_genre: str = None,
):
    if weights is None:
        weights = [0.3, 0.3, 0.4]
    weight_dict = {
        Indexes.CHARACTERS: weights[0],
        Indexes.GENRES: weights[1],
        Indexes.DESCRIPTIONS: weights[2],
    }
    results = search_engine.search(
        query,
        method,
        weight_dict,
        max_results=max_result_count,
        safe_ranking=True,
    )
    if should_print:
        for doc_id, score in results:
            print(doc_id, score)
    return results


def get_book_by_id(id: str, books_dataset: Dict[str, Dict[str, str]]) -> Dict[str, str]:
    return books_dataset.get(str(id), {})
