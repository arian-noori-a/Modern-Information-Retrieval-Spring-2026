import json
from pathlib import Path
from .indexes_enum import Indexes, Index_types
from .index_reader import Index_reader

class Tiered_index:
    def __init__(self, path: str = "index/"):
        self.index = {
            Indexes.CHARACTERS: Index_reader(path, index_name=Indexes.CHARACTERS).index,
            Indexes.GENRES: Index_reader(path, index_name=Indexes.GENRES).index,
            Indexes.DESCRIPTIONS: Index_reader(path, index_name=Indexes.DESCRIPTIONS).index,
        }
        self.tiered_index = {
            Indexes.CHARACTERS: self.convert_to_tiered_index(3, 2, Indexes.CHARACTERS),
            Indexes.DESCRIPTIONS: self.convert_to_tiered_index(10, 5, Indexes.DESCRIPTIONS),
            Indexes.GENRES: self.convert_to_tiered_index(1, 0, Indexes.GENRES),
        }
        self.store_tiered_index(path, Indexes.CHARACTERS)
        self.store_tiered_index(path, Indexes.DESCRIPTIONS)
        self.store_tiered_index(path, Indexes.GENRES)

    def convert_to_tiered_index(self, first_tier_threshold: int, second_tier_threshold: int, index_name):
        if index_name not in self.index:
            raise ValueError("Invalid index type")

        first_tier = {}
        second_tier = {}
        third_tier = {}

        for term, postings in self.index[index_name].items():
            for doc_id, tf in postings.items():
                tf = int(tf)
                if tf >= first_tier_threshold:
                    first_tier.setdefault(term, {})[doc_id] = tf
                elif tf >= second_tier_threshold and second_tier_threshold > 0:
                    second_tier.setdefault(term, {})[doc_id] = tf
                else:
                    third_tier.setdefault(term, {})[doc_id] = tf

        return {
            "first_tier": first_tier,
            "second_tier": second_tier,
            "third_tier": third_tier,
        }

    def store_tiered_index(self, path, index_name):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        file_path = path / f"{index_name.value}_{Index_types.TIERED.value}_index.json"
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.tiered_index[index_name], file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    Tiered_index("../../index/")
