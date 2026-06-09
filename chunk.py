from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []

    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])

        chunks.append((chunk_id, chunk))

        chunk_id += 1
        start += chunk_size - overlap

    return chunks


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    page_id = int(record["page_id"])
    title = record.get("title", "").strip()
    content = record.get("content", "").strip()

    # חיזוק title (חשוב מאוד ל-BM25)
    #full_text = f"{title} {title}\n\n{content}"
    full_text = (
    (title + " ") * 3 +   # BOOST חזק לכותרת
    content
)

    chunks_raw = chunk_text(full_text)

    return [
        Chunk(page_id=page_id, chunk_id=cid, text=text)
        for cid, text in chunks_raw
    ]


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks = []
    for r in records:
        chunks.extend(chunk_entry(r))
    return chunks