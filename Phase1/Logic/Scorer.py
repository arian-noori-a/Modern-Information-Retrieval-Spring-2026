import math
from collections import Counter, defaultdict


class Scorer:
    def __init__(self, index, number_of_documents):
        self.index = index or {}
        self.idf = {}
        self.N = max(int(number_of_documents), 1)
        self._collection_frequencies = None
        self._collection_length = None

    def get_list_of_documents(self, query):
        docs = set()
        for term in query:
            docs.update(self.index.get(term, {}).keys())
        return list(docs)

    def get_idf(self, term):
        if term not in self.idf:
            df = len(self.index.get(term, {}))
            self.idf[term] = 0.0 if df == 0 else math.log10(self.N / df)
        return self.idf[term]

    def get_query_tfs(self, query):
        return Counter(query)

    def compute_scores_with_vector_space_model(self, query, method):
        document_method, query_method = self._parse_vsm_method(method)
        query_tfs = self.get_query_tfs(query)
        scores = {}
        for document_id in self.get_list_of_documents(query):
            score = self.get_vector_space_model_score(query, query_tfs, document_id, document_method, query_method)
            if score != 0:
                scores[document_id] = score
        return scores

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        query_terms = list(dict.fromkeys(query))

        query_weights = {}
        for term in query_terms:
            tf_weight = self._apply_tf(query_tfs.get(term, 0), query_method[0])
            idf_weight = self.get_idf(term) if query_method[1] == "t" else 1.0
            query_weights[term] = tf_weight * idf_weight
        if query_method[2] == "c":
            query_weights = self._cosine_normalize(query_weights)

        document_weights = {}
        for term in query_terms:
            tf = self.index.get(term, {}).get(document_id, 0)
            tf_weight = self._apply_tf(tf, document_method[0])
            idf_weight = self.get_idf(term) if document_method[1] == "t" else 1.0
            document_weights[term] = tf_weight * idf_weight

        if document_method[2] == "c":
            norm = self._document_norm(document_id, document_method)
            if norm > 0:
                document_weights = {term: weight / norm for term, weight in document_weights.items()}

        return sum(query_weights.get(term, 0.0) * document_weights.get(term, 0.0) for term in query_terms)

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        scores = {}
        for document_id in self.get_list_of_documents(query):
            score = self.get_okapi_bm25_score(query, document_id, average_document_field_length, document_lengths)
            if score != 0:
                scores[document_id] = score
        return scores

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        k1 = 1.5
        b = 0.75
        dl = float(document_lengths.get(document_id, 0) or 0)
        avgdl = float(average_document_field_length or 1.0)
        score = 0.0
        for term in query:
            postings = self.index.get(term, {})
            tf = float(postings.get(document_id, 0))
            if tf <= 0:
                continue
            df = len(postings)
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1 - b + b * (dl / avgdl))
            score += idf * ((tf * (k1 + 1)) / denominator)
        return score

    def compute_scores_with_unigram_model(self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5):
        document_lengths = document_lengths or {}
        candidate_docs = self.get_list_of_documents(query) or list(document_lengths.keys())
        scores = {}
        for document_id in candidate_docs:
            score = self.compute_score_with_unigram_model(query, document_id, smoothing_method, document_lengths, alpha, lamda)
            scores[document_id] = score
        return scores

    def compute_score_with_unigram_model(self, query, document_id, smoothing_method, document_lengths, alpha, lamda):
        self._prepare_collection_stats()
        smoothing_method = (smoothing_method or "mixture").lower()
        dl = float(document_lengths.get(document_id, 0) or 0)
        epsilon = 1e-12
        score = 0.0

        for term in query:
            tf = float(self.index.get(term, {}).get(document_id, 0))
            cf = float(self._collection_frequencies.get(term, 0))
            pc = cf / self._collection_length if self._collection_length else epsilon

            if smoothing_method in {"naive", "none"}:
                probability = tf / dl if dl > 0 else epsilon
            elif smoothing_method in {"bayes", "dirichlet"}:
                mu = max(float(alpha), 1e-9)
                probability = (tf + mu * pc) / (dl + mu) if dl + mu > 0 else epsilon
            elif smoothing_method in {"mixture", "linear"}:
                lambda_value = min(max(float(lamda), 0.0), 1.0)
                doc_probability = tf / dl if dl > 0 else 0.0
                probability = lambda_value * doc_probability + (1 - lambda_value) * pc
            else:
                raise ValueError(f"Unknown smoothing method: {smoothing_method}")

            score += math.log(max(probability, epsilon))
        return score

    def _apply_tf(self, tf, mode):
        tf = float(tf or 0)
        if tf <= 0:
            return 0.0
        if mode == "n":
            return tf
        if mode == "l":
            return 1.0 + math.log10(tf)
        if mode == "b":
            return 1.0
        if mode == "a":
            return 0.5 + 0.5 * tf
        return tf

    def _cosine_normalize(self, weights):
        norm = math.sqrt(sum(weight * weight for weight in weights.values()))
        if norm == 0:
            return weights
        return {term: weight / norm for term, weight in weights.items()}

    def _prepare_collection_stats(self):
        if self._collection_frequencies is not None:
            return
        collection_frequencies = Counter()
        for term, postings in self.index.items():
            collection_frequencies[term] = sum(int(tf) for tf in postings.values())
        self._collection_frequencies = collection_frequencies
        self._collection_length = max(sum(collection_frequencies.values()), 1)

    def _parse_vsm_method(self, method):
        method = (method or "ltc.lnc").replace("-", ".").lower()
        if "." not in method:
            return "ltc", "lnc"
        document_method, query_method = method.split(".", 1)
        document_method = (document_method + "nnn")[:3]
        query_method = (query_method + "nnn")[:3]
        return document_method, query_method

    def _document_norm(self, document_id, document_method):
        total = 0.0
        for term, postings in self.index.items():
            tf = postings.get(document_id, 0)
            if tf <= 0:
                continue
            tf_weight = self._apply_tf(tf, document_method[0])
            idf_weight = self.get_idf(term) if document_method[1] == "t" else 1.0
            weight = tf_weight * idf_weight
            total += weight * weight
        return math.sqrt(total)
