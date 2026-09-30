import string
from typing import Callable, List, Tuple


class Snippet:

    def __init__(self, normalize_function: Callable, remove_stopword_function: Callable, number_of_words_on_each_side: int = 5):
        self.number_of_words_on_each_side = number_of_words_on_each_side
        self.normalize = normalize_function
        self.remove_stopword = remove_stopword_function
        self.win_size = (2 * number_of_words_on_each_side) + 1

    def find_snippet(self, raw_doc: str, query: str) -> Tuple[str, List[str]]:
        if raw_doc is None:
            raw_doc = ""
        doc_tokens = str(raw_doc).split()
        if not doc_tokens:
            return "", self.remove_stopword(query)

        query_tokens = self.remove_stopword(query)
        normalized_query = [self.normalize(token) for token in query_tokens]
        query_set = {token for token in normalized_query if token}
        if not query_set:
            return " ".join(doc_tokens[: self.win_size]), []

        normalized_cache = [self.normalize(token.strip(string.punctuation + "“”‘’…—–")) for token in doc_tokens]
        found = set(normalized_cache).intersection(query_set)
        not_exist_words = [query_tokens[i] for i, normalized in enumerate(normalized_query) if normalized and normalized not in found]

        windows = self._identify_best_windows(doc_tokens, normalized_cache, query_set)
        merged_windows = self._merge_windows(windows)
        snippet = self._create_snippet_text(doc_tokens, normalized_cache, merged_windows, query_set)
        return snippet, not_exist_words

    def _identify_best_windows(self, doc_tokens: list, normalized_cache: list, query_set: set) -> List[Tuple[int, int]]:
        if not doc_tokens:
            return []
        scored_windows = []
        n = len(doc_tokens)
        for i, normalized in enumerate(normalized_cache):
            if normalized not in query_set:
                continue
            start = max(0, i - self.number_of_words_on_each_side)
            end = min(n, i + self.number_of_words_on_each_side + 1)
            window_terms = normalized_cache[start:end]
            score = sum(1 for term in window_terms if term in query_set)
            coverage = len(set(window_terms).intersection(query_set))
            scored_windows.append((score, coverage, -(end - start), start, end))

        if not scored_windows:
            return [(0, min(n, self.win_size))]

        scored_windows.sort(reverse=True)
        # Keep a few best windows to balance coverage and compactness.
        selected = []
        covered_terms = set()
        for _, _, _, start, end in scored_windows:
            selected.append((start, end))
            covered_terms.update(set(normalized_cache[start:end]).intersection(query_set))
            if covered_terms >= query_set or len(selected) >= 4:
                break
        return selected

    def _merge_windows(self, windows: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not windows:
            return []
        windows = sorted(windows)
        merged = [windows[0]]
        for start, end in windows[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def _create_snippet_text(self, doc_tokens: list, normalized_cache: list, merged_windows: List[Tuple[int, int]], query_set: set) -> str:
        if not merged_windows:
            return ""
        pieces = []
        for start, end in merged_windows:
            highlighted = []
            for token, normalized in zip(doc_tokens[start:end], normalized_cache[start:end]):
                if normalized in query_set:
                    highlighted.append(f"***{token}***")
                else:
                    highlighted.append(token)
            pieces.append(" ".join(highlighted))
        return " ... ".join(piece for piece in pieces if piece)
