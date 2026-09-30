import os
import pickle
import re
from collections import Counter


class SpellCorrection:
    def __init__(self, all_documents=None, load_path=None, save_path=None):
        if load_path and os.path.exists(load_path):
            self.load(load_path)
        elif all_documents is not None:
            self.all_k_gram_words, self.word_counter = self.k_gramming_and_counting(all_documents)
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                self.save(save_path)
        else:
            self.all_k_gram_words = {}
            self.word_counter = Counter()

    def k_gram_word(self, word, k=2):
        word = str(word).lower().strip()
        if not word:
            return set()
        padded = f"${word}$"
        if len(padded) <= k:
            return {padded}
        return {padded[i:i + k] for i in range(len(padded) - k + 1)}

    def jaccard_score(self, first_set, second_set):
        if not first_set and not second_set:
            return 1.0
        union = first_set.union(second_set)
        if not union:
            return 0.0
        return len(first_set.intersection(second_set)) / len(union)

    def k_gramming_and_counting(self, all_documents):
        word_counter = Counter()
        token_pattern = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
        for document in all_documents:
            if document is None:
                continue
            for match in token_pattern.finditer(str(document).lower()):
                token = match.group(0)
                if len(token) > 1:
                    word_counter[token] += 1
        all_k_gram_words = {word: self.k_gram_word(word) for word in word_counter}
        return all_k_gram_words, word_counter

    def save(self, path):
        data = {
            "all_k_gram_words": self.all_k_gram_words,
            "word_counter": self.word_counter,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.all_k_gram_words = data["all_k_gram_words"]
        self.word_counter = data["word_counter"]

    def find_nearest_words(self, word):
        word = str(word).lower().strip()
        if not word:
            return []
        if word in self.word_counter:
            return [word]

        grams = self.k_gram_word(word)
        if not grams:
            return []
        max_tf = max(self.word_counter.values(), default=1)
        candidates = []
        for vocab_word, vocab_grams in self.all_k_gram_words.items():
            jaccard = self.jaccard_score(grams, vocab_grams)
            if jaccard <= 0:
                continue
            frequency_score = self.word_counter[vocab_word] / max_tf
            score = (0.85 * jaccard) + (0.15 * frequency_score)
            candidates.append((score, jaccard, self.word_counter[vocab_word], vocab_word))
        candidates.sort(reverse=True)
        return [candidate[-1] for candidate in candidates[:5]]

    def spell_check(self, query):
        token_pattern = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
        corrected = []
        last_end = 0
        for match in token_pattern.finditer(str(query)):
            corrected.append(str(query)[last_end:match.start()])
            token = match.group(0)
            lower = token.lower()
            if lower in self.word_counter:
                corrected_token = token
            else:
                nearest = self.find_nearest_words(lower)
                corrected_token = nearest[0] if nearest else token
            corrected.append(corrected_token)
            last_end = match.end()
        corrected.append(str(query)[last_end:])
        return "".join(corrected)
