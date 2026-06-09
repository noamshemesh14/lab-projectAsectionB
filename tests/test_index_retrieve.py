"""
End-to-End Toy Pipeline Test (Plumbing Check).
Run this with: python scripts/test_plumbing.py
"""
import sys
import json
from pathlib import Path
import faiss
import numpy as np

# Ensure Python can find your project modules
STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from embed import embed_queries
from retrieve import search_batch
from utils import ARTIFACTS_DIR

# 1. Create a Fake Dataset 
class DummyChunk:
    def __init__(self, text: str, page_id: int, chunk_id: int):
        self.text = text
        self.page_id = page_id
        self.chunk_id = chunk_id

# Simulated Wikipedia entries with realistic paragraph density and metadata anchors
DUMMY_CHUNKS = [
    DummyChunk(
        "Atmospheric Science | Sky Color\n"
        "The sky appears blue due to a phenomenon called Rayleigh scattering. "
        "This effect involves the scattering of shorter wavelengths of light, such as blue and violet, "
        "by the gases present in Earth's atmosphere much more efficiently than longer wavelengths.", 
        101, 0
    ),
    DummyChunk(
        "Higher Education in Israel | Technion\n"
        "The Technion – Israel Institute of Technology is a leading public research university in Haifa, Israel. "
        "Focusing heavily on engineering and the exact sciences, it is widely recognized as a primary driver "
        "of the nation's high-tech ecosystem and scientific innovations.", 
        102, 1
    ),
    DummyChunk(
        "Artificial Intelligence | Large Language Models\n"
        "Generative AI systems like LLMs rely heavily on deep neural network architectures. "
        "These deep learning frameworks, specifically transformers, allow algorithms to process "
        "and generate human-like text by calculating probabilistic distributions across massive text corpora.", 
        103, 2
    ),
    DummyChunk(
        "Computer Science | Programming Languages\n"
        "Julia is a high-level, high-performance dynamic programming language designed specifically for "
        "numerical and scientific computing. It combines the ease of use of Python with the computational "
        "speed of C, making it highly popular among data scientists and analytics professionals.", 
        104, 3
    ),
    DummyChunk(
        "Botany | Houseplants\n"
        "Syngonium podophyllum, commonly referred to as the Arrowhead Plant, is an evergreen climbing vine. "
        "It requires moderate watering, indirect sunlight, and well-draining soil mixes to prevent "
        "the yellowing of leaves typically observed during colder winter months.", 
        105, 4
    ),
]

def build_toy_artifacts():
    """Simulates your offline index.py phase."""
    print("🛠️  STEP 1: Building Toy Index...")
    
    # Ensure artifacts directory exists
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    texts = [c.text for c in DUMMY_CHUNKS]
    
    # Generate real embeddings for the fake texts
    vectors = embed_queries(texts)
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
        
    # Build and save FAISS index (using L2 normalization for Cosine/Inner Product)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, str(ARTIFACTS_DIR / "index.faiss"))
    
    # Save Meta JSON
    meta = {
        "page_ids": [c.page_id for c in DUMMY_CHUNKS],
        "chunk_ids": [c.chunk_id for c in DUMMY_CHUNKS],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(DUMMY_CHUNKS),
    }
    (ARTIFACTS_DIR / "index_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    # Save Texts JSON (For the Cross-Encoder)
    (ARTIFACTS_DIR / "chunk_texts.json").write_text(json.dumps(texts, indent=2), encoding="utf-8")
    
    print("✅  Artifacts successfully written to disk.\n")


def test_real_retrieval():
    """Simulates your online retrieve.py phase."""
    print("🔍 STEP 2: Testing Retrieval Pipeline...")
    
    # We are looking for the AI architecture chunk (Page ID 103)
    query = "How do large language models work?"
    print(f"   Query: '{query}'")
    
    # Force retrieve.py to reset its global caches in case they are stale
    import retrieve
    retrieve._INDEX = None 
    retrieve._PAGE_IDS = None
    retrieve._CHUNK_TEXTS = None
    retrieve._RERANKER = None
    
    # Run your actual search_batch pipeline!
    results = search_batch([query], top_k=2)
    ranked_page_ids = results[0]
    
    print(f"   Returned Page IDs: {ranked_page_ids}")
    
    if ranked_page_ids and ranked_page_ids[0] == 103:
        print("\n🎉 SUCCESS! The pipeline works perfectly end-to-end.")
        print("   FAISS found the vector, and the Cross-Encoder scored the text correctly.")
    else:
        print("\n❌ FAILURE! The expected Page ID (103) was not ranked first.")


if __name__ == "__main__":
    build_toy_artifacts()
    test_real_retrieval()