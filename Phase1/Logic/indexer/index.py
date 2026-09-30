import copy
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List
from .indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        self.preprocessed_documents = preprocessed_documents
        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.CHARACTERS.value: self.index_characters(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.DESCRIPTIONS.value: self.index_descriptions(),
        }

    @staticmethod
    def _as_tokens(value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [token for token in value.split() if token]
        if isinstance(value, Iterable):
            tokens = []
            for item in value:
                if item is None:
                    continue
                if isinstance(item, str):
                    tokens.extend(token for token in item.split() if token)
                else:
                    tokens.append(str(item))
            return tokens
        return [str(value)]

    def _build_inverted_index(self, field: str) -> Dict[str, Dict[str, int]]:
        inverted_index: Dict[str, Dict[str, int]] = {}
        for document in self.preprocessed_documents:
            doc_id = str(document.get("id"))
            term_counts = Counter(self._as_tokens(document.get(field, [])))
            for term, tf in term_counts.items():
                inverted_index.setdefault(term, {})[doc_id] = int(tf)
        return inverted_index

    def index_documents(self):
        return {str(document["id"]): document for document in self.preprocessed_documents if "id" in document}

    def index_characters(self):
        return self._build_inverted_index(Indexes.CHARACTERS.value)

    def index_genres(self):
        return self._build_inverted_index(Indexes.GENRES.value)

    def index_descriptions(self):
        return self._build_inverted_index(Indexes.DESCRIPTIONS.value)

    def get_posting_list(self, word: str, index_type: str):
        try:
            return list(self.index[index_type].get(word, {}).keys())
        except Exception:
            return []

    def _add_terms_to_index(self, doc_id: str, terms: List[str], index_name: str):
        for term, tf in Counter(terms).items():
            self.index[index_name].setdefault(term, {})[doc_id] = int(tf)

    def add_document_to_index(self, document: dict):
        doc_id = str(document["id"])
        self.index[Indexes.DOCUMENTS.value][doc_id] = document
        self._add_terms_to_index(doc_id, self._as_tokens(document.get(Indexes.CHARACTERS.value, [])), Indexes.CHARACTERS.value)
        self._add_terms_to_index(doc_id, self._as_tokens(document.get(Indexes.GENRES.value, [])), Indexes.GENRES.value)
        self._add_terms_to_index(doc_id, self._as_tokens(document.get(Indexes.DESCRIPTIONS.value, [])), Indexes.DESCRIPTIONS.value)

    def remove_document_from_index(self, document_id: str):
        document_id = str(document_id)
        self.index[Indexes.DOCUMENTS.value].pop(document_id, None)
        for index_name in (Indexes.CHARACTERS.value, Indexes.GENRES.value, Indexes.DESCRIPTIONS.value):
            empty_terms = []
            for term, postings in self.index[index_name].items():
                postings.pop(document_id, None)
                if not postings:
                    empty_terms.append(term)
            for term in empty_terms:
                del self.index[index_name][term]

    def delete_dummy_keys(self, index_before_add, index, key):
        if len(index_before_add[index][key]) == 0:
            del index_before_add[index][key]

    def check_if_key_exists(self, index_before_add, index, key):
        if key not in index_before_add[index]:
            index_before_add[index].setdefault(key, {})

    def check_add_remove_is_correct(self):
        dummy_document = {
            "id": "100",
            "characters": ["sandman", "robin"],
            "genres": ["mystery", "crime"],
            "description": ["good"],
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]["100"] != dummy_document:
            print("Add is incorrect, document")
            return

        for index_name, term in [
            (Indexes.CHARACTERS.value, "sandman"),
            (Indexes.CHARACTERS.value, "robin"),
            (Indexes.GENRES.value, "mystery"),
            (Indexes.GENRES.value, "crime"),
            (Indexes.DESCRIPTIONS.value, "good"),
        ]:
            self.check_if_key_exists(index_before_add, index_name, term)
            if set(index_after_add[index_name][term]).difference(set(index_before_add[index_name][term])) != {dummy_document["id"]}:
                print(f"Add is incorrect, {term}")
                return

        for index_name, term in [
            (Indexes.CHARACTERS.value, "sandman"),
            (Indexes.CHARACTERS.value, "robin"),
            (Indexes.GENRES.value, "mystery"),
            (Indexes.GENRES.value, "crime"),
            (Indexes.DESCRIPTIONS.value, "good"),
        ]:
            self.delete_dummy_keys(index_before_add, index_name, term)

        print("Add is correct")
        self.remove_document_from_index("100")
        index_after_remove = copy.deepcopy(self.index)
        print("Remove is correct" if index_after_remove == index_before_add else "Remove is incorrect")

    def store_index(self, path: str = "index/", index_name: str = None):
        if index_name is None:
            for name in self.index:
                self.store_index(path, name)
            return

        if index_name not in self.index:
            raise ValueError("Invalid index name")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"{index_name}_index.json", "w", encoding="utf-8") as file:
            json.dump(self.index[index_name], file, ensure_ascii=False, indent=2)

    def load_index(self, path: str):
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        print("comparing indexes")
        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = "good"):
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue
            if check_word in self._as_tokens(document[index_type]):
                docs.append(document["id"])
            if len(docs) == 3:
                break
        brute_force_time = time.time() - start

        start = time.time()
        posting_list = self.get_posting_list(check_word, index_type)
        implemented_time = time.time() - start

        print("Brute force time: ", brute_force_time)
        print("Implemented time: ", implemented_time)

        if set(docs).issubset(set(posting_list)):
            print("Indexing is correct")
            if implemented_time < brute_force_time:
                print("Indexing is good")
                return True
            print("Indexing is bad")
            return False
        print("Indexing is wrong")
        return False


def main():
    print("Use build.py from the project root to generate all indexes.")


if __name__ == "__main__":
    main()
