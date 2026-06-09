from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from chunk import chunk_corpus
from bm25 import BM25
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries


INDEX_META = "bm25_meta.json"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
):

    out_dir = artifacts_dir or ensure_artifacts_dir()

    records = list(iter_entries(entries_dir))
    chunks = chunk_corpus(records)

    texts = [c.text.lower().split() for c in chunks]
    page_ids = [c.page_id for c in chunks]

    bm25 = BM25(texts)

    meta = {
        "page_ids": page_ids,
        "texts": [" ".join(t) for t in texts]
    }

    (out_dir / INDEX_META).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )

    return bm25, page_ids, chunks


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[BM25, List[int], List[str]]:

    root = artifacts_dir or ARTIFACTS_DIR

    meta = json.loads((root / INDEX_META).read_text(encoding="utf-8"))

    texts = [t.split() for t in meta["texts"]]
    page_ids = meta["page_ids"]

    bm25 = BM25(texts)

    return bm25, page_ids, meta["texts"]