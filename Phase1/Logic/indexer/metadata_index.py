import json
from pathlib import Path
from .index_reader import Index_reader
from .indexes_enum import Indexes, Index_types


class Metadata_index:
    def __init__(self, path: str = "index/"):
        self.path = Path(path)
        self.documents = self.read_documents(path)
        self.metadata_index = self.create_metadata_index()
        self.store_metadata_index(path)

    def read_documents(self, path):
        return Index_reader(path, Indexes.DOCUMENTS).index

    @staticmethod
    def _field_length(value) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return len(value.split())
        if isinstance(value, list):
            return sum(len(item.split()) if isinstance(item, str) else 1 for item in value if item is not None)
        return 1

    def create_metadata_index(self):
        avg_lengths = {
            Indexes.CHARACTERS.value: self.get_average_document_field_length(Indexes.CHARACTERS.value),
            Indexes.GENRES.value: self.get_average_document_field_length(Indexes.GENRES.value),
            Indexes.DESCRIPTIONS.value: self.get_average_document_field_length(Indexes.DESCRIPTIONS.value),
        }
        return {
            "average_document_length": avg_lengths,
            "document_count": len(self.documents),
        }

    def get_average_document_field_length(self, where):
        if not self.documents:
            return 0.0
        total = sum(self._field_length(document.get(where, [])) for document in self.documents.values())
        return total / len(self.documents)

    def store_metadata_index(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{Indexes.DOCUMENTS.value}_{Index_types.METADATA.value}_index.json"
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.metadata_index, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    Metadata_index("../../index/")
