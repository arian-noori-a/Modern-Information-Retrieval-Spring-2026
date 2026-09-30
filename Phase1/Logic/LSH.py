import itertools
import json
import random
import re
from pathlib import Path

import numpy as np


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        self.documents = documents
        self.num_hashes = int(num_hashes)
        self._all_shingles = None
        self._shingle_sets = None

    def shingle_document(self, document, k=2):
        tokens = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", str(document).lower())
        if not tokens:
            return set()
        if len(tokens) < k:
            return {" ".join(tokens)}
        return {" ".join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}

    def build_characteristic_matrix(self):
        self._shingle_sets = [self.shingle_document(document) for document in self.documents]
        self._all_shingles = sorted(set().union(*self._shingle_sets)) if self._shingle_sets else []
        shingle_to_row = {shingle: idx for idx, shingle in enumerate(self._all_shingles)}
        matrix = np.zeros((len(self._all_shingles), len(self.documents)), dtype=np.uint8)
        for doc_idx, shingles in enumerate(self._shingle_sets):
            for shingle in shingles:
                matrix[shingle_to_row[shingle], doc_idx] = 1
        return matrix

    def min_hash_signature(self):
        characteristic_matrix = self.build_characteristic_matrix()
        num_shingles, num_docs = characteristic_matrix.shape
        if num_shingles == 0 or num_docs == 0:
            return np.empty((self.num_hashes, num_docs), dtype=np.int64)

        prime = self._next_prime(num_shingles * 2 + 1)
        rng = random.Random(42)
        coefficients_a = [rng.randint(1, prime - 1) for _ in range(self.num_hashes)]
        coefficients_b = [rng.randint(0, prime - 1) for _ in range(self.num_hashes)]

        signature = np.full((self.num_hashes, num_docs), np.inf)
        rows = np.arange(num_shingles)
        for hash_idx, (a, b) in enumerate(zip(coefficients_a, coefficients_b)):
            hashed_rows = ((a * rows + b) % prime) % num_shingles
            for doc_idx in range(num_docs):
                present_rows = characteristic_matrix[:, doc_idx].astype(bool)
                if present_rows.any():
                    signature[hash_idx, doc_idx] = hashed_rows[present_rows].min()
        return signature.astype(np.int64)

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        if signature.size == 0:
            return {}
        num_rows, num_docs = signature.shape
        rows_per_band = max(int(rows_per_band), 1)
        bands = min(int(bands), max(num_rows // rows_per_band, 1))
        buckets = {}
        for band in range(bands):
            start = band * rows_per_band
            end = min(start + rows_per_band, num_rows)
            if start >= end:
                break
            for doc_idx in range(num_docs):
                bucket_key = (band, tuple(signature[start:end, doc_idx].tolist()))
                buckets.setdefault(bucket_key, []).append(doc_idx)
        return buckets

    def perform_lsh(self):
        num_bands = min(25, max(self.num_hashes, 1))
        rows_per_band = max(self.num_hashes // num_bands, 1)
        signature = self.min_hash_signature()
        return self.lsh_buckets(signature, num_bands, rows_per_band)

    def jaccard_score(self, first_set, second_set):
        if not first_set and not second_set:
            return 1.0
        union = first_set.union(second_set)
        if not union:
            return 0.0
        return len(first_set.intersection(second_set)) / len(union)

    def jaccard_similarity_test(self, buckets, all_documents):
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1
                    first_doc_id, second_doc_id = comb
                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)
                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)
                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)
                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        if all_near_duplicates == 0:
            print("No near duplicate candidates found.")
        else:
            print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

    @staticmethod
    def _next_prime(number):
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True

        while not is_prime(number):
            number += 1
        return number


def main():
    path = Path(__file__).with_name("LSHFakeData.json")
    if not path.exists():
        print("LSHFakeData.json not found.")
        return
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    documents = []
    for item in data:
        description = item.get("description", item.get("descriptions", ""))
        if isinstance(description, list):
            description = " ".join(description)
        documents.append(description)
    lsh = MinHashLSH(documents, 100)
    buckets = lsh.perform_lsh()
    lsh.jaccard_similarity_test(buckets, documents)


if __name__ == "__main__":
    main()
