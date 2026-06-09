import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np

# Adjust the import paths according to your repository structure
from index import build_index, load_index
from chunk import Chunk


class TestIndexPipeline(unittest.TestCase):
    """
    Unit tests for the offline indexing pipeline and online index loading.
    
    Verifies:
    1. End-to-end index construction from chunked text arrays.
    2. Correct data types (float32) and matrix normalization for Inner Product search.
    3. Metadata JSON serialization and structural alignment (page/chunk ID mapping).
    4. Memory-mapped (MMAP) FAISS index recovery and query execution consistency.
    """

    def setUp(self):
        """Set up temporary directories and dummy data for testing."""
        self.test_dir = Path(__file__).resolve().parent
        self.artifacts_dir = self.test_dir / "temp_artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        # Create mock chunks that simulate what chunk_corpus outputs
        self.mock_chunks = [
            Chunk(page_id=20263, chunk_id=0, text="Point guard basketball finals in 1820s."),
            Chunk(page_id=9112, chunk_id=0, text="Los Angeles basketball franchise captain 1987."),
            Chunk(page_id=9112, chunk_id=1, text="Additional details about the Lakers championship."),
        ]

        # Define vector dimensions matching sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
        self.dimension = 384
        self.num_chunks = len(self.mock_chunks)

        # Generate deterministic mock vectors (un-normalized float32)
        np.random.seed(42)
        self.mock_vectors = np.random.rand(self.num_chunks, self.dimension).astype(np.float32)

    def tearDown(self):
        """Clean up temporary test artifacts from disk."""
        if self.artifacts_dir.exists():
            for file in self.artifacts_dir.iterdir():
                file.unlink()
            self.artifacts_dir.rmdir()

    @patch("index.iter_entries")
    @patch("index.chunk_corpus")
    @patch("index.embed_texts")
    def test_end_to_end_index_pipeline(self, mock_embed, mock_chunk, mock_iter):
        """Test that building, saving, and loading the index maintains exact data alignment."""
        # Arrange mock return values to isolate index.py logic
        mock_iter.return_value = [{"id": "dummy_record"}]  # Minimal dummy list
        mock_chunk.return_value = self.mock_chunks
        mock_embed.return_value = self.mock_vectors.copy()

        # Act: 1. Build the index offline
        built_index, built_page_ids = build_index(
            entries_dir=Path("fake_entries"), 
            artifacts_dir=self.artifacts_dir
        )

        # Assert: Check built artifacts properties
        self.assertIsInstance(built_index, faiss.IndexFlatIP)
        self.assertEqual(len(built_page_ids), self.num_chunks)
        self.assertEqual(built_page_ids, [20263, 9112, 9112])

        # Assert: Verify files actually exist on disk
        faiss_file = self.artifacts_dir / "index.faiss"
        meta_file = self.artifacts_dir / "index_meta.json"
        self.assertTrue(faiss_file.exists())
        self.assertTrue(meta_file.exists())

        # Assert: Verify structural validity of the JSON metadata file
        meta_data = json.loads(meta_file.read_text(encoding="utf-8"))
        self.assertEqual(meta_data["num_vectors"], self.num_chunks)
        self.assertEqual(meta_data["page_ids"], [20263, 9112, 9112])
        self.assertEqual(meta_data["chunk_ids"], [0, 0, 1])
        self.assertEqual(meta_data["model"], "sentence-transformers/all-MiniLM-L6-v2")

        # Act: 2. Load the index (Simulating the online grading setup)
        loaded_index, loaded_page_ids = load_index(artifacts_dir=self.artifacts_dir)

        # Assert: Verify memory-mapped index functions identically
        self.assertEqual(loaded_index.ntotal, self.num_chunks)
        self.assertEqual(loaded_page_ids, built_page_ids)

        # Act: 3. Perform a mock search to verify spatial correctness
        # Let's query using the exact embedding of the second chunk (index 1)
        query_vector = self.mock_vectors[1:2].copy()
        faiss.normalize_L2(query_vector)

        distances, indices = loaded_index.search(query_vector, k=1)
        winning_index = indices[0][0]

        # Assert: The matching FAISS row position must map to page 9112
        self.assertEqual(winning_index, 1)
        self.assertEqual(loaded_page_ids[winning_index], 9112)
        # Cosine similarity of a normalized vector against itself must be extremely close to 1.0
        self.assertAlmostEqual(distances[0][0], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()