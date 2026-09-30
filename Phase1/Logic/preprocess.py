import csv
import json
import math
import re
import string
from pathlib import Path
from typing import Any, Iterable, List
from nltk.stem import PorterStemmer


class Preprocessor:
    def __init__(self, custom_stopwords_path: str = None):

        self.url_email_pattern = re.compile(
            r"\S*http\S*|\S*www\S*|\S+\.ir\S*|\S+\.com\S*|\S+\.org\S*|\S*@\S*",
            flags=re.IGNORECASE,
        )

        self.token_pattern = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

        self.stemmer = PorterStemmer() if PorterStemmer is not None else None

        project_root = Path(__file__).resolve().parents[1]

        if custom_stopwords_path:
            stopwords_path = Path(custom_stopwords_path)
        else:
            stopwords_path = project_root / "stopwords.txt"

        if not stopwords_path.exists():
            raise FileNotFoundError(f"Stopwords file not found: {stopwords_path}")

        with open(stopwords_path, "r", encoding="utf-8") as file:
            self.stopwords = {
                word.strip().lower()
                for word in file
                if word.strip()
            }

    def preprocess_text(self, text: Any) -> List[str]:

        if text is None or (isinstance(text, float) and math.isnan(text)):
            return []

        if isinstance(text, (list, tuple, set)):
            text = " ".join(str(item) for item in text if item is not None)

        text = self.url_email_pattern.sub(" ", str(text))

        tokens = []

        for token in self.remove_stopwords(text):
            normalized_token = self.normalize(token)

            if normalized_token:
                tokens.append(normalized_token)

        return tokens

    def remove_stopwords(self, text: Any) -> List[str]:

        if text is None:
            return []

        tokens = [
            match.group(0).lower()
            for match in self.token_pattern.finditer(str(text))
        ]

        return [
            token
            for token in tokens
            if token and token not in self.stopwords
        ]

    def normalize(self, word: Any) -> str:

        if word is None:
            return ""

        word = str(word).lower().strip(string.punctuation + "“”‘’…—–")

        if not word or word in self.stopwords:
            return ""

        if self.stemmer is not None and word.isalpha():
            return self.stemmer.stem(word)

        return word

    def preprocess_many(self, documents: Iterable[Any]) -> List[List[str]]:

        return [
            self.preprocess_text(document)
            for document in documents
        ]


def _is_empty(value: Any) -> bool:

    return (
        value is None
        or (isinstance(value, float) and math.isnan(value))
        or str(value).strip() == ""
    )


def _clean_text(value: Any) -> str:

    if _is_empty(value):
        return ""

    return str(value).strip()


def _split_csv_list(value: Any) -> List[str]:

    if _is_empty(value):
        return []

    if isinstance(value, list):
        raw_items = value
    else:
        raw_items = str(value).split(",")

    seen = set()
    items = []

    for item in raw_items:
        item = str(item).strip()

        if not item:
            continue

        key = item.lower()

        if key not in seen:
            seen.add(key)
            items.append(item)

    return items


def preprocess_docs(docs: list):
    
    preprocessor = Preprocessor()

    text_fields = [
        "title",
        "author",
        "series",
        "description",
    ]

    list_fields = [
        "genres",
        "awards",
        "characters",
        "places",
        "languages",
    ]

    for doc in docs:
        for field in text_fields:
            doc[field] = preprocessor.preprocess_text(doc.get(field, ""))

        for field in list_fields:
            field_tokens = []

            for item in doc.get(field, []) or []:
                field_tokens.extend(preprocessor.preprocess_text(item))

            doc[field] = field_tokens


def csv_to_json(csv_file_path, json_file_path):

    csv_file_path = Path(csv_file_path)
    json_file_path = Path(json_file_path)

    docs = []

    with open(csv_file_path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            doc = {
                "id": _clean_text(row.get("bookId")),

                "title": _clean_text(row.get("title")),
                "author": _clean_text(row.get("author")),
                "series": _clean_text(row.get("series")),
                "description": _clean_text(row.get("description")),

                "genres": _split_csv_list(row.get("genres")),
                "awards": _split_csv_list(row.get("awards")),
                "characters": _split_csv_list(row.get("characters")),
                "places": _split_csv_list(row.get("places")),
                "languages": _split_csv_list(row.get("language")),

                "first_publish_date": _clean_text(row.get("first_publish_date")),
                "publish_date": _clean_text(row.get("publish_date")),
                "num_pages": _clean_text(row.get("num_pages")),

                "avg_rating": _clean_text(row.get("avg_rating")),
                "num_ratings": _clean_text(row.get("num_ratings")),
                "num_reviews": _clean_text(row.get("num_reviews")),
            }

            if doc["id"]:
                docs.append(doc)

    with open(json_file_path, "w", encoding="utf-8") as json_file:
        json.dump(docs, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]

    csv_path = project_root / "top_3000_rated_books.csv"
    crawled_path = project_root / "crawled.json"
    preprocessed_path = project_root / "preprocessed.json"

    csv_to_json(csv_path, crawled_path)

    with open(crawled_path, "r", encoding="utf-8") as file:
        docs = json.load(file)

    preprocess_docs(docs)

    with open(preprocessed_path, "w", encoding="utf-8") as file:
        json.dump(docs, file, ensure_ascii=False, indent=2)