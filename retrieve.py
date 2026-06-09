from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL


def _minmax(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    mn = float(x.min())
    mx = float(x.max())
    if mx - mn < 1e-9:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:

    bm25, page_ids, _, vectors = load_index(artifacts_dir)

    query_vectors = embed_queries(queries, batch_size=64)

    results = []

    for q, q_vec in zip(queries, query_vectors):
        tokens = q.lower().split()

        candidate_idx, bm25_scores_all = bm25.search(tokens, top_k=200)

        candidate_idx = np.array(candidate_idx, dtype=np.int64)

        bm25_scores = np.array(
            [bm25_scores_all[i] for i in candidate_idx],
            dtype=np.float32,
        )

        emb_scores = vectors[candidate_idx] @ q_vec

        bm25_norm = _minmax(bm25_scores)
        emb_norm = _minmax(emb_scores)

        final_scores = 0.45 * bm25_norm + 0.55 * emb_norm

        order = np.argsort(-final_scores)

        seen = set()
        ranked_pages = []

        for pos in order:
            idx = int(candidate_idx[pos])
            pid = page_ids[idx]

            if pid in seen:
                continue

            seen.add(pid)
            ranked_pages.append(pid)

            if len(ranked_pages) >= top_k:
                break

        results.append(ranked_pages)

    return results