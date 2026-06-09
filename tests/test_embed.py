import unittest
import numpy as np

# Adjust import based on your structure
from embed import embed_texts, embed_queries

class TestEmbeddingPipeline(unittest.TestCase):
    """
    Verifies that the embedding wrapper outputs the exact matrix 
    shapes, types, and norms required by FAISS.
    """

    def test_embedding_shapes_and_types(self):
        # 1. Setup: Two tiny dummy sentences
        texts = ["First dummy sentence.", "Second dummy sentence."]
        
        # 2. Execution
        vectors = embed_texts(texts, batch_size=2)
        
        # 3. Assertions for FAISS compliance
        self.assertEqual(
            vectors.dtype, 
            np.float32, 
            "Vectors must be float32 to prevent FAISS memory crashes."
        )
        self.assertEqual(
            vectors.shape, 
            (2, 384), 
            "Shape mismatch. Expected (2 sentences, 384 MiniLM dimensions)."
        )
        
        # 4. Verify L2 Normalization (Critical for Inner Product = Cosine Similarity)
        # The geometric length (norm) of every vector must be exactly 1.0
        norms = np.linalg.norm(vectors, axis=1)
        for norm in norms:
            self.assertAlmostEqual(
                norm, 
                1.0, 
                places=4, 
                msg="Vectors are not L2 normalized!"
            )

    def test_empty_input_handling(self):
        """Edge case: Ensure the system doesn't crash if passed an empty list."""
        vectors = embed_texts([])
        self.assertEqual(vectors.shape, (0, 384))
        self.assertEqual(vectors.dtype, np.float32)

if __name__ == "__main__":
    unittest.main()