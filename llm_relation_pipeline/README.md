# LLM relation extraction for HSAL-GNN (Goodreads Young Adult)



HSAL-GNN is a graph recommender that uses series information between books, this case study helps if an LLM, using only its own knowledge of a book, tell us whether the book has a **prequel** or a **sequel** in its series, and can the book it names be found among the 8,108 books in our training data? A found book is a candidate new edge for the HSAL graph.

For each of the 8,108 books the pipeline does four things:

1. Ask a local LLM (Llama 3.1 8B or Mistral 7B, through Ollama) whether the book has a prequel (or sequel) and which book it is.
2. Match the book the LLM names to a book in the corpus: exact title first, then cosine similarity on sentence embeddings.
3. Build a ground truth from the series information already in the dataset (`r_sequel`).
4. Score the LLM's yes/no answers against that ground truth (precision, recall, F1, specificity).

Two variants are compared. **Version A** prompts with the title only. **Version B** prompts with title and author, and its embeddings include the author name. Version B did better on Llama in the full-corpus run, so it is the one to start from.

 The reports (`reports/HSAL_LLM_Relation_Enhancement_Report_v4.pdf` and `.docx`) have the full results and the caveats; read sections 5, 13 and 14 before relying on any number.

## Folder layout

```
llm_relation_pipeline/
  inputs/        everything the pipeline reads
    Goodreads_young_adult_HSAL_100k.csv    HSAL interaction file
    metadata/                              one row per book, Version A and Version B
    embeddings/version_A/, version_B/      sentence embeddings, item index, embedded text
  samples/       the frozen 400-book and 30-book evaluation sets
  src/           all Python code (run it from anywhere, paths come from src/paths.py)
  outputs/       everything the pipeline writes
    ground_truth/   extraction/   matches/   scores/
  reports/       the written report (v4, pdf and docx)
  analysis/      diagnostic script and files for the relation-edge analysis
  ccr/           Slurm scripts for the UB CCR cluster
  upstream/      Colab cells that built inputs/ (provenance only)
```

| Path | What it is |
|---|---|
| `inputs/Goodreads_young_adult_HSAL_100k.csv` | The HSAL interaction file: `user_id, item_id, time, item_series_id` (99,922 rows, 56,619 users, 8,108 items). `item_series_id` is `Series_<id>_P<n>` for a book at position n of a series, or `SB<id>` for a standalone book. |
| `inputs/metadata/Goodreads_young_adult_item_meta_with_title.csv` | Version A metadata: `item_id, title, i_genre_c, r_sequel`. `r_sequel` lists every later book of the same series that is in the corpus. |
| `inputs/metadata/Goodreads_young_adult_author.meta.csv` | Version B metadata: adds `description, authors, average_rating, ratings_count`. |
| `inputs/embeddings/version_A/Goodreads_young_adult.{feat.npy,item2index,text}` | Version A embeddings (8108 x 384), the item_id to row index, and the exact text that was embedded. |
| `inputs/embeddings/version_B/Goodreads_young_adult_author.{feat.npy,item2index,text}` | The same for Version B. |
| `samples/prequel_sample_400.csv`, `samples/sequel_sample_400.csv` | The 400-book samples (165 positive, 165 negative, 70 ambiguous). Frozen inputs. |
| `samples/prequel_eval_set_30.csv`, `samples/sequel_eval_set_30.csv` | The earlier 30-book evaluation sets. |
| `src/paths.py` | Every file location in one place. |
| `src/ollama_client.py` | One shared function that calls Ollama (temperature 0.2, JSON output, 2 retries). |
| `src/matching_core.py` | Shared matching logic: title normalisation, series checks, series-position rerank, coherence check, threshold 0.5. |
| `src/extract_prequel.py`, `src/extract_sequel.py` | LLM extraction. |
| `src/match_prequel.py`, `src/match_sequel.py` | Matching the LLM's answers to corpus books. |
| `src/build_ground_truth.py` | Builds the ground truth from `r_sequel`. |
| `src/score_full_corpus.py`, `src/score_sample400.py`, `src/score_sequel400.py`, `src/score_sequel30.py` | Scoring. |
| `outputs/ground_truth/ground_truth_full_corpus.csv` | Ground truth for all 8,108 books. |
| `outputs/extraction/prequel/`, `outputs/extraction/sequel/` | Raw LLM outputs: `*_vA_*` and `*_vB_*` are the full corpus, `*_sample400_*` are the 400-book runs, `*_eval30_*` the 30-book runs. |
| `outputs/matches/full_corpus/`, `outputs/matches/sample400/` | Matching outputs. |
| `outputs/scores/full_corpus_scoring_results.csv` | Scoring output for the 8 full-corpus runs. |

## Install

The Mac runs used Python 3.13. Python 3.10 or newer should work, but only 3.13 was used with these pinned versions.

```
cd llm_relation_pipeline
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

`sentence-transformers` downloads the `all-MiniLM-L6-v2` model from Hugging Face the first time the matching step runs.

For the extraction step you also need [Ollama](https://ollama.com) running, with both models pulled:

```
ollama pull llama3.1:8b
ollama pull mistral
```

## Raw data (only needed to rebuild `inputs/` and the embeddings)

You do not need these to run the pipeline. The 100k interaction file, the metadata and the embeddings are all committed. The raw files come from the UCSD Goodreads dataset (Wan and McAuley, RecSys 2018; Wan et al., ACL 2019),  Download page: https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/

| File | Used for |
|---|---|
| `byGenre/goodreads_books_young_adult.json.gz` (105 MB) | Titles, descriptions, genres, author ids for the 8,108 books. |
| `goodreads_book_authors.json.gz` (18 MB) | Author id to author name (Version B). |
| `byGenre/goodreads_interactions_young_adult.json.gz` | Only to rebuild the 100k interaction file. |
| `goodreads_books.json.gz` | Only to rebuild the franchise map used for series positions. |



## How the committed inputs were made

`upstream/build_hsal_dataset_colab_cells.py` has the Colab cells that produced `inputs/Goodreads_young_adult_HSAL_100k.csv`, and `upstream/build_embeddings_colab_cells.py` has the cells that produced the metadata and embeddings

Both embedding versions use `all-MiniLM-L6-v2` (384 dimensions). The embedded text is `title + description (first 500 chars) + genre` for Version A and `title + description + genre + author` for Version B, where genre is the first Goodreads popular shelf that is not on a skip list of generic shelves.


## Execution order

1. **Ground truth.** A book has a sequel if its `r_sequel` is non-empty. It has a prequel if its id appears in another book's `r_sequel`. This gives 3,665 positives for each relation.
   ```
   python3 src/build_ground_truth.py
   ```
2. **LLM extraction.** One call per book. `--eval` runs the 400-book sample instead of all 8,108 books. Use `--limit 20` to test first. Runs resume if interrupted. The full-corpus runs took about 31 hours of Slurm time in total on CCR with 4 concurrent requests.
   ```
   python3 src/extract_sequel.py  --version B --model llama3.1:8b --workers 4
   python3 src/extract_prequel.py --version B --model llama3.1:8b --workers 4
   ```
   Output: `outputs/extraction/sequel/sequels_vB_llama3.1_8b_verified.csv` (and the prequel equivalent).
3. **Matching.** Needs `sentence-transformers`. Exact title match first (rejected if every candidate is confirmed to be in a different series), then cosine similarity, with a series-position rerank among the top 5 within 3 positions, and a cosine threshold of 0.5. For the 400-book samples:
   ```
   python3 src/match_sequel.py --version B
   ```
   For a full-corpus extraction file:
   ```
   python3 src/match_sequel.py --version B --test-file outputs/extraction/sequel/sequels_vB_llama3.1_8b_verified.csv --test-label llama3.1_8b
   ```
   The label becomes part of the output filename, written to `outputs/matches/full_corpus/`. Some committed files use other labels, so check the existing names before comparing.
4. **Scoring.**
   ```
   python3 src/score_full_corpus.py          # all 8 full-corpus runs -> outputs/scores/full_corpus_scoring_results.csv
   python3 src/score_sample400.py --version B    # prequel, 400-book sample
   python3 src/score_sequel400.py                # sequel, 400-book sample
   ```

## Reproducing the results

| Level | What to run | Result |
|---|---|---|
| 1. Ground truth and scoring | Step 1 and step 4 on the committed extraction files. No LLM, no GPU, seconds. | . |
| 2. Matching | Step 3 on the committed extraction files. | Should match if the same embedding model is used.  |
| 3. LLM extraction | Step 2. | |
| 4. Rebuild the inputs | `upstream/` plus the raw files. |  |

## CCR (UB cluster)

`ccr/run_*.sh` start Ollama in an Apptainer container on a scavenger GPU node and run the extraction for one relation, version and model, they are specific to UB CCR. Override the defaults with environment variables, create the log folder, and submit from this folder (the scripts call `src/extract_*.py`):

```
mkdir -p slurm_logs
REPO_DIR=$PWD OLLAMA_SIF=/path/to/ollama.sif sbatch ccr/run_sequel_llama_B.sh
```

