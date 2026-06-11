import math
import numpy as np
from collections import Counter, defaultdict
from typing import List, Tuple

class BM25:
    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.N = len(corpus)
        
        # Precalculate all document lengths and penalties once (Saves time in search)
        self.doc_len = np.array([len(doc) for doc in corpus], dtype=np.float32)
        self.avgdl = float(np.mean(self.doc_len)) if self.N > 0 else 1.0
        
        self.doc_penalty = self.k1 * (1 - self.b + self.b * (self.doc_len / self.avgdl))
        
        self.idf = {}
        # Stores parallel NumPy arrays: (doc_ids_array, term_freqs_array)
        self.inverted_index = {}
        
        self._build(corpus)

    def _build(self, corpus: List[List[str]]) -> None:
        temp_index = defaultdict(list)
        
        for doc_id, doc in enumerate(corpus):
            term_counts = Counter(doc)
            for term, tf in term_counts.items():
                temp_index[term].append((doc_id, tf))
        
        for term, postings in temp_index.items():
            df = len(postings)
            self.idf[term] = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            
            # Convert standard lists into ultra-fast NumPy arrays
            doc_ids = np.array([p[0] for p in postings], dtype=np.int32)
            tfs = np.array([p[1] for p in postings], dtype=np.float32)
            
            self.inverted_index[term] = (doc_ids, tfs)

    def search(self, query_tokens: List[str], top_k: int) -> Tuple[List[int], List[float]]:
        scores = np.zeros(self.N, dtype=np.float32)
        k1_plus_1 = self.k1 + 1.0
        
        # Deduplicate query tokens to prevent doing the exact same math twice (Your idea)
        # query_terms = set(query_tokens)
        
        for q in query_tokens:
            if q not in self.inverted_index:
                continue
                
            doc_ids, tfs = self.inverted_index[q]
            idf = self.idf[q]
            
            # Highly optimized vectorized BM25 math
            denoms = tfs + self.doc_penalty[doc_ids]
            term_scores = idf * (tfs * k1_plus_1) / denoms
            
            # Instantly maps and adds scores to the correct documents
            scores[doc_ids] += term_scores
            
        top_n = min(top_k, self.N)
        if top_n == 0:
            return [], []
            
        # Fast Top-K extraction without sorting the whole array
        if top_n >= self.N:
            ranked_arr = np.argsort(-scores)
            final_scores_arr = scores[ranked_arr]
        else:
            kth = top_n - 1
            partition_idx = np.argpartition(-scores, kth)[:top_n]
            
            top_scores = scores[partition_idx]
            sorted_local_idx = np.argsort(-top_scores)
            
            ranked_arr = partition_idx[sorted_local_idx]
            final_scores_arr = top_scores[sorted_local_idx]
        
        # Filter out absolute zero scores (documents that matched nothing)
        valid_mask = final_scores_arr > 0
        
        ranked = ranked_arr[valid_mask].tolist()
        final_scores = final_scores_arr[valid_mask].tolist()
        
        return ranked, final_scores