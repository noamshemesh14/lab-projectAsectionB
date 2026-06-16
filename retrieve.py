"""Competition-level Query-time retrieval with Cross Encoder reranking."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from utils import K_EVAL, normalize_page_id, rrf, cross_score

def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:

    bm25, page_ids, raw_texts, corpus_vectors = load_index(artifacts_dir)

    query_vectors = embed_queries(queries)

    if query_vectors.size == 0 or corpus_vectors.size == 0:
        return [[] for _ in queries]

    # normalize embeddings (VERY IMPORTANT)
    query_vectors = query_vectors / (np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-9)
    corpus_vectors = corpus_vectors / (np.linalg.norm(corpus_vectors, axis=1, keepdims=True) + 1e-9)

    dense_scores_all = query_vectors @ corpus_vectors.T

    max_idx = corpus_vectors.shape[0]

    # stronger candidate pools
    BM25_POOL = min(1000, max_idx)
    DENSE_POOL = min(1000, max_idx)

    RERANK_POOL = 80

    results = []

    for i, q in enumerate(queries):

        tokens = q.lower().split()
        dense_scores = dense_scores_all[i]

        # ---------------- BM25 ----------------
        bm25_candidates, bm25_scores = bm25.search(tokens, top_k=BM25_POOL)
        bm25_candidates = np.array(bm25_candidates, dtype=np.int64)

        # ---------------- Dense ----------------
        dense_candidates = np.argpartition(-dense_scores, DENSE_POOL)[:DENSE_POOL]

        # ---------------- Ranks ----------------
        bm25_rank = np.argsort(np.argsort(-np.array(bm25_scores)))
        dense_rank = np.argsort(np.argsort(-dense_scores[dense_candidates]))

        # ---------------- RRF fusion ----------------
        bm25_map = {cid: rrf(np.array([r]))[0] for cid, r in zip(bm25_candidates, bm25_rank)}

        fusion_scores = np.zeros(len(dense_candidates))

        for j, idx in enumerate(dense_candidates):
            bm25_val = bm25_map.get(idx, 0.0)
            dense_val = rrf(np.array([dense_rank[j]]))[0]

            fusion_scores[j] = 0.5 * bm25_val + 0.5 * dense_val

        # ---------------- top candidates ----------------
        top_local = np.argpartition(-fusion_scores, RERANK_POOL)[:RERANK_POOL]
        rerank_idx = dense_candidates[top_local]

        rerank_docs = [raw_texts[int(idx)] for idx in rerank_idx]

        # ---------------- Cross Encoder ----------------
        ce_scores = cross_score(q, rerank_docs)
        ce_scores = np.array(ce_scores)

        # ---------------- final ranking ----------------
        order = np.argsort(-ce_scores)

        seen = set()
        out = []

        for pos in order:
            pid = normalize_page_id(page_ids[int(rerank_idx[pos])])

            if pid in seen:
                continue

            seen.add(pid)
            out.append(pid)

            if len(out) >= top_k:
                break

        results.append(out)

    return results