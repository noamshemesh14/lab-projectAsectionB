# Section B — Retrieval Pipeline

A high-performance information retrieval system that combines **BM25 sparse retrieval**, **dense embeddings**, and **cross-encoder reranking** to rank Wikipedia documents against search queries.

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

## 📋 File Descriptions

### Core Modules

#### **main.py**
- **Purpose**: Autograder interface
- **Key Function**: `run(queries: List[str]) -> List[List[int]]`
  - Takes a batch of queries (e.g., 50 at grading time)
  - Returns ranked page IDs per query (top-10 scored)
- **Also**: `build_offline_index()` – creates `artifacts/` locally (not timed)

#### **retrieve.py** ⭐ Main Retrieval Logic
- **Key Function**: `search_batch(queries)` – orchestrates the entire ranking pipeline
- **Process Flow**:
  1. **Load indexes** from `artifacts/`
  2. **Embed queries** using sentence-transformers
  3. **BM25 candidate pool** (top 1000)
  4. **Dense candidate pool** (top 1000 by cosine similarity)
  5. **RRF fusion** (combine BM25 + dense ranks with 0.3/0.7 weighting)
  6. **De-duplication** by page_id (keep highest score per page)
  7. **Cross-encoder reranking** (top 150 candidates)
  8. **Final ranking** by cross-encoder score (top 10 unique pages)

#### **index.py**
- **Key Functions**:
  - `build_index()` – offline index creation
    - Chunks corpus
    - Trains BM25
    - Generates dense embeddings
    - Saves `artifacts/`
  - `load_index()` – loads all artifacts at inference time
- **Artifacts**:
  - `bm25_index.pkl` – pickled BM25 object
  - `chunk_vectors.npy` – float32 embeddings (L2-normalized)
  - `bm25_meta.json` – page IDs and text for lookup

#### **chunk.py**
- **Key Function**: `chunk_entry(record)` – breaks Wikipedia pages into overlapping chunks
- **Parameters**:
  - `chunk_size=120` words per chunk
  - `overlap=30` words between chunks
- **Feature**: Title boosting (title repeated 3x for higher BM25 scores)
- **Output**: List of `Chunk` objects (page_id, chunk_id, text)

#### **embed.py**
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Key Function**: `embed_texts(texts)` → L2-normalized float32 embeddings (384-dim)
- **Features**:
  - Auto-detects GPU (CUDA) / CPU
  - Batch processing (64-sized batches)
  - Lazy loading (loads model once globally)

#### **bm25.py** ⚡ Optimized BM25
- **Algorithm**: Standard BM25 (Okapi variant)
- **Parameters**: `k1=1.5`, `b=0.75`
- **Optimizations**:
  - Pre-computed document penalties (saves repeated calculations)
  - NumPy vectorized scoring (fast bulk operations)
  - Fast top-K extraction (partition-based, avoids full sort)
- **Key Method**: `search(query_tokens, top_k)` → ranked doc IDs + scores

#### **utils.py**
- **Paths**: `DATA_DIR`, `ENTRIES_DIR`, `ARTIFACTS_DIR`, `PUBLIC_QUERIES_PATH`
- **Helpers**:
  - `normalize_page_id()` – coerce JSON page IDs to int
  - `iter_entries()` – load corpus from JSON files
  - `ensure_artifacts_dir()` – create `artifacts/`
- **Fusion & Scoring**:
  - `rrf(rank)` – Reciprocal Rank Fusion: `1 / (60 + rank)`
  - `cross_score(query, docs)` – use cross-encoder model on pairs
  - `get_ce()` – lazy-load `cross-encoder/ms-marco-MiniLM-L6-v2`

#### **eval.py** (Read-Only)
- **Metric**: NDCG@10 (Normalized Discounted Cumulative Gain)
- **Key Functions**:
  - `ndcg_at_k()` – compute single query NDCG
  - `mean_ndcg_at_k()` – average across all queries
  - `evaluate_run()` – run model and return scores

---

## 🔄 Work Process / Pipeline

### Phase 1: Offline (Local, Not Timed)
```
python main.py
  ↓
build_offline_index()
  ↓
1. Corpus Loading: iter_entries() loads all JSON files from data/Wikipedia Entries/
2. Chunking: chunk_corpus() breaks each page into 120-word chunks (overlap=30)
3. BM25 Training: BM25() builds inverted index + IDF scores
4. Dense Embedding: embed_texts() generates 384-dim vectors for all chunks
5. Artifact Export: Save to artifacts/
   - bm25_index.pkl (BM25 object)
   - chunk_vectors.npy (embeddings)
   - bm25_meta.json (page IDs, texts)
```

### Phase 2: Query-Time Retrieval (Timed at Grading)
```
run(queries: List[str])
  ↓
search_batch(queries)
  ↓
1. Load artifacts from disk (index.py → load_index())
2. Embed queries using sentence-transformers (embed.py)
3. For each query:
   a) Tokenize query (lowercase, split)
   b) BM25 search → top 1000 candidate chunks
   c) Dense search → top 1000 by cosine similarity
   d) Rank fusion (RRF) → combine BM25 + dense ranks
   e) De-duplicate → keep top chunk per page
   f) Cross-encoder rerank → top 150
   g) Final rank → top 10 unique page IDs
4. Return List[List[int]] – one ranked list per query
```

### Phase 3: Evaluation
```
python scripts/eval_public.py
  ↓
- Load public queries + ground truth
- Call run(queries)
- Compute mean NDCG@10
- Display results
```

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

**Expected time**: ~5-10 min (depends on corpus size & GPU availability)

### 3. Test Locally
```bash
python scripts/eval_public.py
```
Evaluates on public query set and prints **mean NDCG@10**.

---

## 🧠 Key Design Decisions

### 1. **Hybrid Retrieval (BM25 + Dense)**
- **BM25**: Exact keyword matching, fast, interpretable
- **Dense embeddings**: Semantic similarity, catches synonyms & paraphrases
- **RRF fusion**: Combines both without manual weighting tweaks

### 2. **Title Boosting**
- Title repeated 3× in chunks → higher BM25 scores for title matches
- Improves recall on page-level relevance

### 3. **Cross-Encoder Reranking**
- BM25 + dense give ~150 candidates
- Cross-encoder (`ms-marco-MiniLM-L6-v2`) scores pairs (query, doc) **jointly**
- Better accuracy than separate embedding scores, small computational cost

### 4. **Page-Level Deduplication**
- Chunks belong to multiple pages
- Top chunk per page is kept, others discarded
- Avoids ranking the same page multiple times

### 5. **Vectorized BM25**
- NumPy arrays for document penalties, term frequencies
- Batch scoring without Python loops
- ~10x faster than naive implementation

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

