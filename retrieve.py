from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from index import load_index
from utils import K_EVAL


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:

    bm25, page_ids, _ = load_index(artifacts_dir)

    results = []

    for q in queries:
        tokens = q.lower().split()

        ranked_idx, _ = bm25.search(tokens, top_k=50)

        seen = set()
        ranked_pages = []

        for idx in ranked_idx:
            pid = page_ids[idx]

            if pid in seen:
                continue
            seen.add(pid)
            ranked_pages.append(pid)

            if len(ranked_pages) >= top_k:
                break

        results.append(ranked_pages)

    return results