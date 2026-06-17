# Section B — Retrieval Pipeline

A high-performance information retrieval system that combines **BM25 sparse retrieval**, **dense embeddings**, and **cross-encoder reranking** to rank Wikipedia documents against search queries.

---

## link to the presentation video: 

https://drive.google.com/file/d/18nlgSR6UxZ3_Wjtm5I0gYi3ueoZc5QTd/view?usp=sharing

---

## 🎯 Project Overview

This project implements a **hybrid retrieval pipeline** designed for rapid, accurate document ranking. The system:

1. **Preprocesses** the corpus into chunks
2. **Builds indexes** for both BM25 (sparse) and dense (semantic) retrieval
3. **Retrieves candidates** using dual pathways
4. **Fuses rankings** via Reciprocal Rank Fusion (RRF)
5. **Reranks** top candidates using a cross-encoder model

**Goal**: Maximize NDCG@10 (Normalized Discounted Cumulative Gain at top-10 results) on hidden test queries.

---

## 📂 Project Structure

```
lab-projectAsectionB/
├── main.py              # Entry point (autograder interface)
├── requirements.txt     # Python dependencies
├── README.md           # This file
│
├── retrieve.py         # Query-time retrieval & ranking
├── embed.py            # Embedding generation (sentence-transformers)
├── index.py            # Index building & loading
├── chunk.py            # Text chunking with overlap
├── bm25.py             # BM25 ranking algorithm (vectorized)
├── utils.py            # Shared paths, helpers, cross-encoder
├── eval.py             # Evaluation metrics (NDCG@10)
│
├── artifacts/          # Built indexes (auto-generated)
│   ├── bm25_index.pkl        # Serialized BM25 object
│   ├── chunk_vectors.npy      # Dense embeddings (NumPy)
│   └── bm25_meta.json         # Metadata (page IDs, texts)
│
├── data/
│   ├── Wikipedia Entries/     # Corpus JSON files (one per page)
│   └── public_queries.json    # Test query set
│
└── scripts/
    ├── build_index.py         # Offline index building
    └── eval_public.py         # Public evaluation script
```

---

## 🔄 Work Process / Pipeline

#### **Phase 1: Offline Index Building**
- **Step 1: Corpus Loading** – Loads raw JSON files from the data directory.
- **Step 2: Text Chunking** – Breaks pages into 120-word segments (30-word overlap), repeating the title 3 times at the heading of each chunk.
- **Step 3: Sparse Indexing** – Builds the BM25 inverted index and calculates IDF scores.
- **Step 4: Dense Embedding** – Computes 384-dimensional embeddings using `all-MiniLM-L6-v2` for all chunks.
- **Step 5: Artifact Export** – Serializes and saves the BM25 object, embeddings array, and metadata to local storage.

#### **Phase 2: Query-Time Retrieval**
- **Step 1: Initialization** – Loads indexes and metadata into memory.
- **Step 2: Query Embedding** – Converts incoming queries into L2-normalized dense vectors.
- **Step 3: Dense Scoring** – Computes cosine similarity (dot product) between queries and all corpus vectors.
- **Step 4: Initial Pooling** – Fetches the top 1000 BM25 candidates and top 1000 Dense candidates.
- **Step 5: Dense-Driven Fusion (RRF)** – Iterates only through the 1000 dense candidates to calculate rank fusion (0.7 Dense + 0.3 BM25). BM25-only candidates are discarded.
- **Step 6: Rerank Pooling** – Selects the top 150 chunks based on the calculated fusion scores.
- **Step 7: Pre-Rerank De-duplication** – Filters the top 150 chunks to retain only the single highest-scoring chunk per unique page ID, efficiently preventing redundant cross-encoder calculations on the same page.
- **Step 8: Cross-Encoder Scoring** – Passes the deduplicated chunks through a cross-encoder for final pair-wise similarity scoring.
- **Step 9: Final Output** – Sorts by cross-encoder score, applies a secondary safety check to ensure absolutely no duplicate pages remain, and returns the top 10 unique page IDs.

#### **Phase 3: Evaluation & Scoring**
- **Metric Calculation** – Computes mean NDCG@10 across all queries.

---

## 🚀 Quick Start

### 1. Setup
```bash
cd path/to/student
pip install -r requirements.txt
```

**Dependencies**:
- `numpy>=1.24` – NumPy arrays & operations
- `sentence-transformers>=2.2.0` – embedding model + cross-encoder
- `faiss-cpu>=1.7.4` – (optional; not currently used)

### 2. Build Index (Offline)
```bash
python main.py
```
This generates `artifacts/` with:
- `bm25_index.pkl`
- `chunk_vectors.npy`
- `bm25_meta.json`

**Expected time**: ~30 mins using GPU for embedding

### 3. Test Locally
```bash
python scripts/eval_public.py
```
Evaluates on public query set and prints **mean NDCG@10**.

---

## 📊 Hyperparameters

| Module | Parameter | Value | Purpose |
|--------|-----------|-------|---------|
| `chunk.py` | `chunk_size` | 120 | Words per chunk |
| `chunk.py` | `overlap` | 30 | Overlap between chunks |
| `embed.py` | `batch_size` | 64 | Embedding batch size |
| `bm25.py` | `k1` | 1.5 | BM25 saturation parameter |
| `bm25.py` | `b` | 0.75 | BM25 length normalization |
| `retrieve.py` | `BM25_POOL` | 1000 | BM25 candidate pool |
| `retrieve.py` | `DENSE_POOL` | 1000 | Dense candidate pool |
| `retrieve.py` | `RERANK_POOL` | 150 | Cross-encoder candidates |
| `retrieve.py` | RRF weights | 0.3 / 0.7 | BM25 / Dense in fusion |
| `utils.py` | `K_EVAL` | 10 | Final ranking depth (scored at grading) |

