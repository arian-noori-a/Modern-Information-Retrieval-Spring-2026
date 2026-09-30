import json
from pathlib import Path
from typing import Any, Iterable

from .indexes_enum import Indexes, Index_types
from .index_reader import Index_reader


class DocumentLengthsIndex:
    def __init__(self, path: str = "index/"):
        self.path = Path(path)
        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        self.document_length_index = {
            Indexes.CHARACTERS: self.get_documents_length(Indexes.CHARACTERS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.DESCRIPTIONS: self.get_documents_length(Indexes.DESCRIPTIONS.value),
        }
        self.store_document_lengths_index(path, Indexes.CHARACTERS)
        self.store_document_lengths_index(path, Indexes.GENRES)
        self.store_document_lengths_index(path, Indexes.DESCRIPTIONS)

    @staticmethod
    def _field_length(value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len([token for token in value.split() if token])
        if isinstance(value, Iterable):
            total = 0
            for item in value:
                if isinstance(item, str):
                    total += len([token for token in item.split() if token])
                elif item is not None:
                    total += 1
            return total
        return 1

    def get_documents_length(self, where):
        return {
            str(doc_id): self._field_length(document.get(where, []))
            for doc_id, document in self.documents_index.items()
        }

    def store_document_lengths_index(self, path, index_name):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{index_name.value}_{Index_types.DOCUMENT_LENGTH.value}_index.json"
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.document_length_index[index_name], file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    DocumentLengthsIndex("../../index/")
    print("Document lengths index stored successfully.")
