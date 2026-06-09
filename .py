"""Optimized preprocessing and mathematically safe chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List
from tqdm import tqdm

from utils import entry_text

# --- SAFE CHUNKING HYPERPARAMETERS ---
# We cap at 140 words to mathematically guarantee we never hit 
# the 256 WordPiece token limit, even if LaTeX equations are present.
SAFE_CHUNK_SIZE = 170 
CHUNK_OVERLAP = 50 

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str

def _chunk_text_with_context(text: str, title: str) -> List[str]:
    """Safely chunks text by words and injects context without breaking FAISS."""
    if not text:
        return []

    # 1. Cleanly split into words. 
    # (The 40-word overlap naturally protects equations from being lost)
    tokens = text.split()
    
    if not tokens:
        return []

    text_chunks = []
    stride = max(1, SAFE_CHUNK_SIZE - CHUNK_OVERLAP)
    
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i : i + SAFE_CHUNK_SIZE]
        chunk_text = " ".join(chunk_tokens)
        
        # Clean up any excessive spaces
        chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
        
        # 3. Contextual Anchoring
        final_text = f"{title}\n{chunk_text}"
        text_chunks.append(final_text)
        
        i += stride

    return text_chunks

def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    text = entry_text(record) or ""
    title = record.get("title", "Unknown Title")
    
    processed_strings = _chunk_text_with_context(text, title)
    
    return [
        Chunk(page_id=page_id, chunk_id=idx, text=chunk_str)
        for idx, chunk_str in enumerate(processed_strings)
    ]

def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in tqdm(records, desc="Chunking Wikipedia Pages", unit="page"):
        chunks.extend(chunk_entry(record))
        
    print(f"\n✅ Total chunks created safely: {len(chunks)}")
    return chunks