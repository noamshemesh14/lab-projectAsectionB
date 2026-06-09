"""
Exploratory Data Analysis for Text Corpus with Visualizations.
Run this with: python scripts/explore_corpus.py
"""
import json
import re
from pathlib import Path
import pandas as pd

# --- VM FIX: Force Matplotlib to run in headless background mode ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

# Updated to point to the directory containing your individual Wikipedia JSON files
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "Wikipedia Entries"

def load_corpus(directory_path: Path) -> pd.DataFrame:
    """Loads the raw corpus from a directory of JSON files into a pandas DataFrame."""
    print(f"Loading data from directory: {directory_path}...")
    records = []
    
    # Find all .json files in the specified directory
    json_files = list(directory_path.glob("*.json"))
    
    if not json_files:
        print(f"⚠️ Warning: No .json files found inside {directory_path}")
        return pd.DataFrame()
        
    for filepath in json_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                
                # Unify the format: make it a list whether it's a single dict or multiple
                items = data if isinstance(data, list) else [data]
                
                for item in items:
                    if isinstance(item, dict):
                        # Explicitly map your JSON schema to the script's schema
                        records.append({
                            "page_id": item.get("page_id", ""),
                            "title": item.get("title", ""),
                            "text": item.get("content", "")  
                        })
                        
        except json.JSONDecodeError:
            print(f"⚠️ Error: Could not parse JSON in {filepath.name}")
            
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} documents from {len(json_files)} files.")
    
    # Safety check: ensure the DataFrame has a 'text' column for the profiler
    if not df.empty and 'text' not in df.columns:
        print("⚠️ Warning: Your JSON files do not have a 'text' key. The profiler will fail.")
        
    return df

def profile_text_lengths(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates word counts and their mathematical distribution."""
    print("\n--- Length Distribution (Word Count) ---")
    
    # Quick whitespace split for a rough word count
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    # Standard statistical summary
    desc = df['word_count'].describe(percentiles=[.25, .5, .75, .90, .95, .99])
    
    print(f"Mean size:   {desc['mean']:.0f} words")
    print(f"Median size: {desc['50%']:.0f} words")
    print(f"75th %ile:   {desc['75%']:.0f} words")
    print(f"90th %ile:   {desc['90%']:.0f} words")
    print(f"99th %ile:   {desc['99%']:.0f} words")
    print(f"Max size:    {desc['max']:.0f} words")
    
    return df

def analyze_structure(df: pd.DataFrame) -> dict:
    """Detects Markdown, hierarchical headers, and special characters."""
    print("\n--- Structural Analysis ---")
    
    # Detect Markdown Headers
    df['has_h1'] = df['text'].str.contains(r'^#\s+', flags=re.MULTILINE, regex=True)
    df['has_h2_h3'] = df['text'].str.contains(r'^#{2,3}\s+', flags=re.MULTILINE, regex=True)
    
    # Detect Markdown Links or Image tags
    df['has_links'] = df['text'].str.contains(r'\[.*?\]\(.*?\)', regex=True)
    
    # Detect LaTeX math blocks or inline math
    df['has_latex'] = df['text'].str.contains(r'\$\$.*?\$\$|\$.*?\$', regex=True)
    
    total = len(df)
    stats = {
        'Primary Headers (H1)': (df['has_h1'].sum() / total) * 100,
        'Subheaders (H2/H3)': (df['has_h2_h3'].sum() / total) * 100,
        'Markdown Links': (df['has_links'].sum() / total) * 100,
        'LaTeX/Math': (df['has_latex'].sum() / total) * 100
    }
    
    for key, value in stats.items():
        print(f"Docs with {key}: {value:.1f}%")
        
    return stats

def plot_analysis(df: pd.DataFrame, structure_stats: dict):
    """Generates and saves EDA plots without opening a window."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Word Count Distribution (Histogram)
    cap_val = df['word_count'].quantile(0.95)
    filtered_df = df[df['word_count'] <= cap_val]
    
    sns.histplot(filtered_df['word_count'], bins=50, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title("Word Count Distribution (Capped at 95th Percentile)")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Number of Documents")
    
    # Plot 2: Structural Elements (Bar Chart)
    labels = list(structure_stats.keys())
    values = list(structure_stats.values())
    
    sns.barplot(x=labels, y=values, ax=axes[1], palette="viridis")
    axes[1].set_title("Prevalence of Structural Elements")
    axes[1].set_ylabel("Percentage of Documents (%)")
    axes[1].set_ylim(0, 100)
    
    for i, v in enumerate(values):
        axes[1].text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    
    analytics_dir = PROJECT_ROOT / "analytics"    
    analytics_dir.mkdir(exist_ok=True)
    out_path = analytics_dir / "corpus_eda_plots.png"
    
    # --- VM FIX: Save the plot and close the figure ---
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plots successfully saved to {out_path}")
    plt.close(fig)

def run_profiler():
    if not RAW_DATA_PATH.exists() or not RAW_DATA_PATH.is_dir():
        print(f"❌ Error: Could not find directory at {RAW_DATA_PATH}")
        return
        
    df = load_corpus(RAW_DATA_PATH)
    if df.empty:
        return
        
    df = profile_text_lengths(df)
    stats = analyze_structure(df)
    plot_analysis(df, stats)
    print("\n✅ Profiling complete. Ready for remote run.")

if __name__ == "__main__":
    run_profiler()