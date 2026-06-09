import math
from collections import defaultdict, Counter
from typing import List


class BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        self.N = len(corpus)
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / self.N if self.N > 0 else 0

        self.df = defaultdict(int)
        self.idf = {}

        self._build()

    def _build(self):
        for doc in self.corpus:
            seen = set(doc)
            for term in seen:
                self.df[term] += 1

        for term, freq in self.df.items():
            self.idf[term] = math.log(
                1 + (self.N - freq + 0.5) / (freq + 0.5)
            )

    def score(self, query_tokens: List[str], index: int) -> float:
        doc = self.corpus[index]
        tf = Counter(doc)

        doc_len = self.doc_len[index]
        score = 0.0

        for q in query_tokens:
            if q not in tf:
                continue

            f = tf[q]
            idf = self.idf.get(q, 0.0)

            denom = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (f * (self.k1 + 1)) / denom

        return score

    def search(self, query_tokens: List[str], top_k: int):
        scores = [
            self.score(query_tokens, i)
            for i in range(self.N)
        ]

        ranked = sorted(
            range(self.N),
            key=lambda i: scores[i],
            reverse=True
        )

        return ranked[:top_k], scores