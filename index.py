from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from chunk import chunk_corpus
from bm25 import BM25
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, iter_entries

INDEX_META = "bm25_meta.json"
VECTORS_FILE = "chunk_vectors.npy"
BM25_FILE = "bm25_index.pkl"


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
):
    out_dir = artifacts_dir or ensure_artifacts_dir()

    records = list(iter_entries(entries_dir))
    chunks = chunk_corpus(records)

    raw_texts = [c.text for c in chunks]
    tokenized_texts = [t.lower().split() for t in raw_texts]
    page_ids = [c.page_id for c in chunks]

    bm25 = BM25(tokenized_texts)

    with open(out_dir / BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)

    vectors = embed_texts(raw_texts, batch_size=64)
    np.save(out_dir / VECTORS_FILE, vectors)

    meta = {
        "page_ids": page_ids,
        "texts": raw_texts,
    }

    (out_dir / INDEX_META).write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8"
    )

    return bm25, page_ids, raw_texts, vectors


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[BM25, List[int], List[str], np.ndarray]:
    root = artifacts_dir or ARTIFACTS_DIR

    meta = json.loads((root / INDEX_META).read_text(encoding="utf-8"))

    raw_texts = meta["texts"]
    page_ids = meta["page_ids"]

    with open(root / BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    vectors = np.load(root / VECTORS_FILE)

    return bm25, page_ids, raw_texts, vectors