import math
from typing import List


class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        self._validate(actual, predicted)
        values = []
        for relevant, retrieved in zip(actual, predicted):
            retrieved_set = set(retrieved)
            if not retrieved_set:
                values.append(0.0)
            else:
                values.append(len(set(relevant).intersection(retrieved_set)) / len(retrieved_set))
        return sum(values) / len(values) if values else 0.0

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        self._validate(actual, predicted)
        values = []
        for relevant, retrieved in zip(actual, predicted):
            relevant_set = set(relevant)
            if not relevant_set:
                values.append(0.0)
            else:
                values.append(len(relevant_set.intersection(set(retrieved))) / len(relevant_set))
        return sum(values) / len(values) if values else 0.0

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        relevant = set(actual)
        if not relevant:
            return 0.0
        hits = 0
        precision_sum = 0.0
        seen = set()
        for rank, doc_id in enumerate(predicted, start=1):
            if doc_id in relevant and doc_id not in seen:
                hits += 1
                precision_sum += hits / rank
                seen.add(doc_id)
        return precision_sum / len(relevant)

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        self._validate(actual, predicted)
        values = [self._average_precision_single(a, p) for a, p in zip(actual, predicted)]
        return sum(values) / len(values) if values else 0.0

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        return self.calculate_AP(actual, predicted)

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        relevant = set(actual)
        dcg = 0.0
        seen = set()
        for rank, doc_id in enumerate(predicted, start=1):
            if doc_id in relevant and doc_id not in seen:
                dcg += 1.0 / math.log2(rank + 1)
                seen.add(doc_id)
        return dcg

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        self._validate(actual, predicted)
        values = [self._dcg_single(a, p) for a, p in zip(actual, predicted)]
        return sum(values) / len(values) if values else 0.0

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        self._validate(actual, predicted)
        values = []
        for relevant, retrieved in zip(actual, predicted):
            dcg = self._dcg_single(relevant, retrieved)
            ideal = list(relevant)[: len(retrieved)] if retrieved else list(relevant)
            idcg = self._dcg_single(relevant, ideal)
            values.append(0.0 if idcg == 0 else dcg / idcg)
        return sum(values) / len(values) if values else 0.0

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        self._validate(actual, predicted)
        values = []
        for relevant, retrieved in zip(actual, predicted):
            relevant_set = set(relevant)
            rr = 0.0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant_set:
                    rr = 1.0 / rank
                    break
            values.append(rr)
        return sum(values) / len(values) if values else 0.0

    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        return self.cacluate_RR(actual, predicted)

    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        print(f"name = {self.name}")
        print(f"Precision = {precision:.6f}")
        print(f"Recall = {recall:.6f}")
        print(f"F1 = {f1:.6f}")
        print(f"AP = {ap:.6f}")
        print(f"MAP = {map:.6f}")
        print(f"DCG = {dcg:.6f}")
        print(f"NDCG = {ndcg:.6f}")
        print(f"RR = {rr:.6f}")
        print(f"MRR = {mrr:.6f}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "ap": ap,
                    "map": map,
                    "dcg": dcg,
                    "ndcg": ndcg,
                    "rr": rr,
                    "mrr": mrr,
                })
        except Exception:
            pass

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "ap": ap,
            "map": map_score,
            "dcg": dcg,
            "ndcg": ndcg,
            "rr": rr,
            "mrr": mrr,
        }
