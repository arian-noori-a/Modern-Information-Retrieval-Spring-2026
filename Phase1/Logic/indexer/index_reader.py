import json
from pathlib import Path
from .indexes_enum import Indexes, Index_types

class Index_reader:
    def __init__(self, path: str = "index/", index_name: Indexes = None, index_type: Index_types = None):
        self.path = Path(path)
        self.index_name = index_name
        self.index_type = index_type
        self.index = self.get_index()

    def get_index(self):
        if self.index_name is None:
            raise ValueError("index_name is required")

        file_name = self.index_name.value
        if self.index_type is not None:
            file_name += f"_{self.index_type.value}"
        file_name += "_index.json"

        absolute_path = self.path / file_name
        with open(absolute_path, "r", encoding="utf-8") as file:
            return json.load(file)
