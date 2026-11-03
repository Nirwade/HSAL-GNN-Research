"""
Single place for every file location used by the scripts in this folder.
Paths are resolved from this file's location, so the scripts work from any working directory.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

META_A = ROOT / "inputs" / "metadata" / "Goodreads_young_adult_item_meta_with_title.csv"
META_B = ROOT / "inputs" / "metadata" / "Goodreads_young_adult_author.meta.csv"

# Embedding prefixes: append ".feat.npy", ".item2index" or ".text"
EMB_A = ROOT / "inputs" / "embeddings" / "version_A" / "Goodreads_young_adult"
EMB_B = ROOT / "inputs" / "embeddings" / "version_B" / "Goodreads_young_adult_author"

SAMPLES = ROOT / "samples"
GROUND_TRUTH_DIR = ROOT / "outputs" / "ground_truth"
EXTRACTION_PREQUEL = ROOT / "outputs" / "extraction" / "prequel"
EXTRACTION_SEQUEL = ROOT / "outputs" / "extraction" / "sequel"
MATCHES_FULL = ROOT / "outputs" / "matches" / "full_corpus"
MATCHES_SAMPLE400 = ROOT / "outputs" / "matches" / "sample400"
SCORES_DIR = ROOT / "outputs" / "scores"
