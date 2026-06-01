"""Optional preprocessing and chunking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from utils import entry_text

CHUNK_SIZE = 300
CHUNK_OVERLAP = 60

@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str

def split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()

    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]

        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if end >= len(words):
            break

    return chunks


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry into retrieval units.

    Strategy:
    - Fixed-size word chunks
    - Overlapping windows
    - Title included in every chunk
    """

    page_id = int(record["page_id"])

    title = record.get("title", "")
    text = entry_text(record)

    raw_chunks = split_text(
        text=text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    chunks = []

    for idx, chunk_text in enumerate(raw_chunks):
        full_text = f"{title}\n\n{chunk_text}"

        chunks.append(
            Chunk(
                page_id=page_id,
                chunk_id=idx,
                text=full_text,
            )
        )

    return chunks

def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []

    for idx, record in enumerate(records):
        if idx % 1000 == 0:
            print(f"Chunked {idx}/{len(records)} pages")

        chunks.extend(chunk_entry(record))

    print(f"Total chunks created: {len(chunks)}")

    return chunks

def main() -> None:
    import json
    from pathlib import Path
    import numpy as np

    data_dir = Path("data/Wikipedia Entries")
    files = sorted(data_dir.glob("*.json"))

    print(f"Found {len(files)} files")

    records = []
    for file_path in files[:100]:
        with open(file_path, "r", encoding="utf-8") as f:
            records.append(json.load(f))

    chunks = chunk_corpus(records)
    lengths = np.array([len(chunk.text.split()) for chunk in chunks])

    print()
    print("===== CHUNK TEST =====")
    print("pages checked:", len(records))
    print("chunks created:", len(chunks))
    print("avg chunks per page:", round(len(chunks) / len(records), 2))
    print()
    print("chunk length stats:")
    print("min:", int(np.min(lengths)))
    print("max:", int(np.max(lengths)))
    print("mean:", round(float(np.mean(lengths)), 2))
    print("median:", round(float(np.median(lengths)), 2))
    print("p95:", round(float(np.percentile(lengths, 95)), 2))

    print()
    print("===== EXAMPLE CHUNKS =====")
    for chunk in chunks[:3]:
        print()
        print(f"page_id={chunk.page_id}, chunk_id={chunk.chunk_id}")
        print(chunk.text[:700])
        print("-" * 80)

        print()
        print("===== OVERLAP CHECK =====")

        c0 = chunks[0].text.split()
        c1 = chunks[1].text.split()

    print()
    print("Last 20 words of chunk 0:")
    print(" ".join(c0[-20:]))

    print()
    print("First 20 words of chunk 1:")
    print(" ".join(c1[:20]))

    print()
    print("===== OVERLAP CHECK =====")

    same_page_chunks = [c for c in chunks if c.page_id == chunks[0].page_id]

    c0_words = same_page_chunks[0].text.split()
    c1_words = same_page_chunks[1].text.split()

    # remove title words from the beginning
    title_words = records[0].get("title", "").split()
    title_len = len(title_words)

    c0_body = c0_words[title_len:]
    c1_body = c1_words[title_len:]

    print()
    print("Last 30 body words of chunk 0:")
    print(" ".join(c0_body[-30:]))

    print()
    print("First 30 body words of chunk 1:")
    print(" ".join(c1_body[:30]))

    print()
    print("Overlap equal?")
    print(c0_body[-30:] == c1_body[:30])

if __name__ == "__main__":
    main()