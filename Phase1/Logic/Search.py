from .preprocess import Preprocessor
from .Scorer import Scorer
from .indexer import Indexes, Index_types, Index_reader

class SearchEngine:
    def __init__(self, path="index/"):
        self.path = path
        self.preprocessor = Preprocessor()
        self.fields = [Indexes.CHARACTERS, Indexes.GENRES, Indexes.DESCRIPTIONS]

        self.document_indexes = {
            Indexes.CHARACTERS: Index_reader(path, Indexes.CHARACTERS).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES).index,
            Indexes.DESCRIPTIONS: Index_reader(path, Indexes.DESCRIPTIONS).index,
        }

        self.tiered_index = {}
        for field in self.fields:
            try:
                self.tiered_index[field] = Index_reader(path, field, Index_types.TIERED).index
            except Exception:
                self.tiered_index[field] = {"first_tier": self.document_indexes[field], "second_tier": {}, "third_tier": {}}

        self.document_lengths_index = {
            Indexes.CHARACTERS: Index_reader(path, Indexes.CHARACTERS, Index_types.DOCUMENT_LENGTH).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH).index,
            Indexes.DESCRIPTIONS: Index_reader(path, Indexes.DESCRIPTIONS, Index_types.DOCUMENT_LENGTH).index,
        }

        self.documents_index = Index_reader(path, Indexes.DOCUMENTS).index

        try:
            self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA).index
        except Exception:
            self.metadata_index = {
                "document_count": len(self.documents_index),
                "average_document_length": {
                    field.value: (
                        sum(self.document_lengths_index[field].values()) / len(self.document_lengths_index[field])
                        if len(self.document_lengths_index[field]) > 0 else 0.0
                    )
                    for field in self.fields
                },
            }

    def search(self, query, method, weights, safe_ranking=True, max_results=10, smoothing_method=None, alpha=0.5, lamda=0.5):
        processed_query = query if isinstance(query, list) else self.preprocessor.preprocess_text(query)
        processed_query = [term for term in processed_query if term]
        if not processed_query:
            return []

        scores = {}
        method_normalized = (method or "ltc.lnc").lower()

        if "unigram" in method_normalized:
            self.find_scores_with_unigram_model(processed_query, smoothing_method or "mixture", weights, scores, alpha, lamda)
        elif method_normalized in {"okapibm25", "bm25", "okapi"}:
            self.find_scores_with_safe_ranking(processed_query, "OkapiBM25", weights, scores)
        elif safe_ranking:
            self.find_scores_with_safe_ranking(processed_query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(processed_query, method, weights, max_results, scores)

        final_scores = {}
        self.aggregate_scores(weights, scores, final_scores)
        ranked = sorted(final_scores.items(), key=lambda item: item[1], reverse=True)
        if max_results is not None and max_results != -1:
            ranked = ranked[: int(max_results)]
        return ranked

    def aggregate_scores(self, weights, scores, final_scores):
        for field, weight in weights.items():
            if weight == 0 or field not in scores:
                continue
            for doc_id, score in scores[field].items():
                final_scores[doc_id] = final_scores.get(doc_id, 0.0) + weight * score

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        tier_names = ["first_tier", "second_tier", "third_tier"]
        for field in self.fields:
            if weights.get(field, 0) == 0:
                continue
            merged_index = {}
            field_scores = {}
            for tier_name in tier_names:
                for term, postings in self.tiered_index[field].get(tier_name, {}).items():
                    merged_index.setdefault(term, {}).update(postings)
                scorer = Scorer(merged_index, len(self.documents_index))
                if (method or "").lower() in {"okapibm25", "bm25", "okapi"}:
                    field_scores = scorer.compute_socres_with_okapi_bm25(
                        query, self._get_average_length(field), self.document_lengths_index[field]
                    )
                else:
                    field_scores = scorer.compute_scores_with_vector_space_model(query, method)
                if max_results == -1 or len(field_scores) >= max_results:
                    break
            scores[field] = field_scores

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        for field in self.fields:
            if weights.get(field, 0) == 0:
                continue
            scorer = Scorer(self.document_indexes[field], len(self.documents_index))
            if (method or "").lower() in {"okapibm25", "bm25", "okapi"}:
                scores[field] = scorer.compute_socres_with_okapi_bm25(
                    query, self._get_average_length(field), self.document_lengths_index[field]
                )
            else:
                scores[field] = scorer.compute_scores_with_vector_space_model(query, method)

    def find_scores_with_unigram_model(self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5):
        for field in self.fields:
            if weights.get(field, 0) == 0:
                continue
            scorer = Scorer(self.document_indexes[field], len(self.documents_index))
            scores[field] = scorer.compute_scores_with_unigram_model(
                query, smoothing_method, self.document_lengths_index[field], alpha, lamda
            )

    def merge_scores(self, scores1, scores2):
        merged = dict(scores1)
        for doc_id, score in scores2.items():
            merged[doc_id] = merged.get(doc_id, 0.0) + score
        return merged

    def _get_average_length(self, field):
        avg_lengths = self.metadata_index.get("average_document_length", {})
        if field.value in avg_lengths:
            return avg_lengths[field.value]
        lengths = self.document_lengths_index[field]
        return sum(lengths.values()) / len(lengths) if len(lengths) > 0 else 0.0


if __name__ == "__main__":
    search_engine = SearchEngine()
    weights = {Indexes.CHARACTERS: 1, Indexes.GENRES: 1, Indexes.DESCRIPTIONS: 1}
    print(search_engine.search("magic adventure", "ltc.lnc", weights))
