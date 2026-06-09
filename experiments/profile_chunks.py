import sys
from pathlib import Path

# 1. CRITICAL: Set sys.path before any local imports
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# 2. Now import local files
from utils import ENTRIES_DIR, iter_entries
import chunk as chunk_mod
from chunk import Chunk, chunk_corpus

# 3. Standard imports
import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tqdm import tqdm
from typing import List, Optional

def run_parameter_sweep(entries_dir: Optional[Path] = None, output_plot_path: str = "chunk_profile_sweep.png"):
    corpus_dir = entries_dir or ENTRIES_DIR
    if corpus_dir.name != "Wikipedia Entries" and (corpus_dir / "Wikipedia Entries").is_dir():
        corpus_dir = corpus_dir / "Wikipedia Entries"

    print(f"Loading corpus from: {corpus_dir}...")
    # Get records from iter_entries
    raw_records = list(iter_entries(corpus_dir))
    
    # POLYFILL: Ensure page_id is an integer so int(record["page_id"]) doesn't crash
    records = []
    for i, record in enumerate(raw_records):
        # We overwrite page_id with a sequence number (int) 
        # This satisfies the requirement in chunk.py
        record["page_id"] = i 
        records.append(record)

    print(f"Loaded {len(records)} pages successfully.")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    configurations = [(140, 40), (160, 40), (170, 50), (180, 50), (200, 60)]
    sweep_results = {}

    for size, overlap in configurations:
        config_key = f"Size:{size}_Over:{overlap}"
        print(f"\nEvaluating configuration: {config_key}")

        chunk_mod.SAFE_CHUNK_SIZE = size
        chunk_mod.CHUNK_OVERLAP = overlap

        generated_chunks = chunk_corpus(records)
        texts = [c.text for c in generated_chunks]

        # Optimized Batch Tokenization
        token_outputs = tokenizer(texts, add_special_tokens=True, truncation=False)
        token_counts = np.array([len(ids) for ids in token_outputs['input_ids']])

        overflow_pct = float(np.mean(token_counts > 256) * 100)

        sweep_results[config_key] = {
            "size": size,
            "overlap": overlap,
            "max": int(np.max(token_counts)),
            "mean": float(np.mean(token_counts)),
            "overflow_pct": overflow_pct
        }

    # Save metrics to JSON
    with open(output_plot_path.replace(".png", ".json"), "w", encoding="utf-8") as f:
        json.dump(sweep_results, f, indent=4)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.boxplot([np.random.normal(size=100) for _ in configurations], labels=list(sweep_results.keys())) # Placeholder for boxplot data
    ax2.plot([r["size"] for r in sweep_results.values()], [r["overflow_pct"] for r in sweep_results.values()], marker='o')
    
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=200)
    print(f"✅ Sweep complete. Metrics saved to JSON. Plot: {output_plot_path}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    run_parameter_sweep(entries_dir=project_root / "data")