"""
Comprehensive Sanity Check Sandbox for Chunking Logic.
Run this with: python tests/test_chunking.py
"""
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Bulletproof pathing to find your root project files dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Adjust 'chunk' to match the exact name of your python file (e.g., chunk.py or chunking.py)
from chunk import chunk_corpus, CHUNK_SIZE, CHUNK_OVERLAP, Chunk

def load_all_records(data_dir: Path) -> List[Dict[str, Any]]:
    """Loads the ENTIRE JSON corpus into memory."""
    assert data_dir.exists(), f"❌ Error: Could not find {data_dir}"
    files = sorted(data_dir.glob("*.json"))
    assert len(files) > 0, "❌ Error: No JSON files found."

    records = []
    # Wrap the loop in tqdm so the terminal doesn't freeze while loading 27,000 files
    for file_path in tqdm(files, desc="Loading JSON files to memory", unit="file"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    # Pass raw dict so utils.entry_text works natively
                    records.append(item)
                    
    return records

def load_sample_records(data_dir: Path, sample_size: int = 50) -> List[Dict[str, Any]]:
    """Loads a small subset of the JSON corpus into memory for rapid testing."""
    assert data_dir.exists(), f"❌ Error: Could not find {data_dir}"
    files = sorted(data_dir.glob("*.json"))
    assert len(files) > 0, "❌ Error: No JSON files found."

    records = []
    for file_path in files[:sample_size]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            items = data if isinstance(data, list) else [data]
            for item in items:
                if isinstance(item, dict):
                    records.append({
                        "page_id": item.get("page_id", ""),
                        "title": item.get("title", ""),
                        "text": item.get("content", "")
                    })
    return records


def print_chunk_metrics(records: List[Dict[str, Any]], chunks: List[Chunk]) -> None:
    """Calculates and prints the statistical distribution of the chunked text."""
    lengths = np.array([len(chunk.text.split()) for chunk in chunks])
    
    num_pages = len(records)
    num_chunks = len(chunks)
    avg_chunks_per_page = round(num_chunks / num_pages, 2)
    
    print("\n" + "="*30)
    print("📊 CHUNK METRICS SUMMARY")
    print("="*30)
    print(f"Pages processed:       {num_pages}")
    print(f"Total chunks created:  {num_chunks}")
    print(f"Avg chunks per page:   {avg_chunks_per_page}")
    print("-" * 30)
    print("Length stats (in words):")
    print(f"  Min:    {int(np.min(lengths))}")
    print(f"  Max:    {int(np.max(lengths))} (Expected slightly > {CHUNK_SIZE} due to title anchoring)")
    print(f"  Mean:   {round(float(np.mean(lengths)), 2)}")
    print(f"  Median: {round(float(np.median(lengths)), 2)}")
    print(f"  P95:    {round(float(np.percentile(lengths, 95)), 2)}")
    print("="*30 + "\n")


def assert_schema_and_boundaries(chunks: List[Chunk]) -> None:
    """Validates that every chunk meets schema rules, size limits, and formatting."""
    print("Running basic schema and boundary assertions...")
    
    for chunk in chunks:
        # Schema Verification
        assert hasattr(chunk, 'page_id'), "Schema Error: Missing 'page_id'"
        assert hasattr(chunk, 'chunk_id'), "Schema Error: Missing 'chunk_id'"
        assert hasattr(chunk, 'text'), "Schema Error: Missing 'text'"
        
        words = chunk.text.split()
        assert len(words) > 0, f"Data Error: Chunk {chunk.chunk_id} on page {chunk.page_id} is empty."
        
        # Upper Bound Check (Allowing 1.5x buffer for unbroken math equations)
        assert len(words) <= CHUNK_SIZE * 1.5, (
            f"Sizing Error: Chunk {chunk.chunk_id} on page {chunk.page_id} has {len(words)} words, "
            f"exceeding the CHUNK_SIZE limit."
        )

        # Context Anchor Check
        assert chunk.text.startswith("Title: "), (
            f"Context Error: Chunk {chunk.chunk_id} on page {chunk.page_id} missing 'Title: ' prefix."
        )

        # LaTeX Parity Check
        dollar_count = chunk.text.count('$')
        assert dollar_count % 2 == 0, (
            f"LaTeX Error: Chunk {chunk.chunk_id} on page {chunk.page_id} "
            f"has an unbalanced number of $ symbols ({dollar_count})."
        )


def assert_sequence_and_overlap(chunks: List[Chunk]) -> None:
    """Validates that chunks from the same page increment sequentially and overlap exactly."""
    print("Running sequence and overlap assertions...")
    
    chunks_by_page = {}
    for chunk in chunks:
        chunks_by_page.setdefault(chunk.page_id, []).append(chunk)

    for page_id, page_chunks in chunks_by_page.items():
        page_chunks.sort(key=lambda c: c.chunk_id)
        
        # 1. Sequence Verification
        for idx, chunk in enumerate(page_chunks):
            assert chunk.chunk_id == idx, (
                f"Sequence Error on page {page_id}: Expected chunk_id {idx}, found {chunk.chunk_id}."
            )
        
        if len(page_chunks) < 2:
            continue
            
        # 2. Overlap Verification
        for i in range(len(page_chunks) - 1):
            c0_words = page_chunks[i].text.split()
            c1_words = page_chunks[i+1].text.split()
            
            # Dynamically calculate the title length to strip it out before comparing overlap
            title_prefix_len = len(page_chunks[0].text.split('\n')[0].split())
            
            c0_body = c0_words[title_prefix_len:]
            c1_body = c1_words[title_prefix_len:]
            
            # Only check overlap if both chunks are larger than the overlap amount
            if len(c0_body) >= CHUNK_OVERLAP and len(c1_body) >= CHUNK_OVERLAP:
                trailing_window = c0_body[-CHUNK_OVERLAP:]
                leading_window = c1_body[:CHUNK_OVERLAP]
                
                assert trailing_window == leading_window, (
                    f"Overlap Error on page {page_id}:\n"
                    f"Trailing window: {' '.join(trailing_window)}\n"
                    f"Leading window: {' '.join(leading_window)}"
                )


def run_all_tests() -> None:
    """Main orchestrator that ties the test suite together."""
    data_dir = PROJECT_ROOT / "data" / "Wikipedia Entries"
    
    # 1. Load Data
    records = load_all_records(data_dir)
    
    # 2. Execute Chunking
    print(f"Running chunker (Target Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP})...")
    chunks = chunk_corpus(records)
    assert len(chunks) > 0, "❌ Assertion Failed: chunk_corpus returned an empty list"

    # 3. Run the Suite
    print_chunk_metrics(records, chunks)
    assert_schema_and_boundaries(chunks)
    assert_sequence_and_overlap(chunks)

    print("\n✅ All Assertions Passed. Chunking logic is structurally sound and production-ready.")


if __name__ == "__main__":
    run_all_tests()