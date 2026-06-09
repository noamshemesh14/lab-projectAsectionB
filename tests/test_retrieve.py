import unittest
import numpy as np

# Adjust import based on your project structure
import retrieve


class TestRetrievalMegaBatchLogic(unittest.TestCase):
    
    def setUp(self):
        """
        Runs before every test. We inject fake data directly into 
        the retrieve.py global variables, completely bypassing disk I/O.
        """
        # Chunk 0 and Chunk 1 both belong to Page 10.
        # Chunk 2 belongs to Page 20.
        retrieve._PAGE_IDS = [10, 10, 20] 
        retrieve._CHUNK_TEXTS = [
            "Page 10 Text Part 1", 
            "Page 10 Text Part 2", 
            "Page 20 Text"
        ]

    def test_build_mega_batch_skips_invalid_offsets(self):
        """FAISS returns -1 when it cannot find enough neighbors. Ensure we skip them entirely."""
        # Query 0 gets a valid chunk and two -1s. Query 1 gets a valid chunk.
        queries = ["Query A", "Query B"]
        fake_faiss_indices = np.array([
            [2, -1, -1],
            [0, -1, -1]
        ])
        
        all_pairs, counts, offsets = retrieve._build_mega_batch(queries, fake_faiss_indices)
        
        # Verify text pairs built correctly (ignoring -1s)
        self.assertEqual(all_pairs, [("Query A", "Page 20 Text"), ("Query B", "Page 10 Text Part 1")])
        # Verify counts only track valid pairs
        self.assertEqual(counts, [1, 1])
        # Verify tracking offsets match
        self.assertEqual(offsets, [[2], [0]])

    def test_process_mega_batch_sorting_and_deduplication(self):
        """
        Tests if the algorithm correctly splits the 1D score array back into 
        individual queries, sorts by highest score, and deduplicates pages.
        """
        # Setup: Simulate 1 single query that matched 3 chunks (offsets 0, 1, 2)
        query_pair_counts = [3]
        valid_offsets_per_query = [[0, 1, 2]]
        
        # Simulate Cross-Encoder scores for those 3 chunks
        # Chunk 0 -> Score 0.0 (Page 10)
        # Chunk 1 -> Score 5.0 (Page 10)
        # Chunk 2 -> Score 10.0 (Page 20)
        fake_scores = np.array([0.0, 5.0, 10.0])
        
        results = retrieve._process_mega_batch_results(
            all_scores=fake_scores,
            query_pair_counts=query_pair_counts,
            valid_offsets_per_query=valid_offsets_per_query,
            top_k=2
        )
        
        # Expected behavior:
        # - Highest score is 10.0 (Chunk 2 -> Page 20)
        # - Next highest is 5.0 (Chunk 1 -> Page 10)
        # - Lowest is 0.0 (Chunk 0 -> Page 10, which should be skipped because Page 10 was already seen)
        # Output layout: List of lists (one per query) -> [[20, 10]]
        
        self.assertEqual(results, [[20, 10]])

    def test_empty_query_batch(self):
        """Ensure the main loop doesn't crash if given an empty list of queries."""
        self.assertEqual(
            retrieve.search_batch([]), 
            [], 
            "Pipeline should return an empty list for empty queries."
        )


if __name__ == "__main__":
    unittest.main()