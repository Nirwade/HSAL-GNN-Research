# Chunked HSAL and HSAL+LRD: Anime and ML-1M

Chunked graph construction (8 user-disjoint chunks) compared with the original single-file pipeline on two more
datasets, with the same settings as the Goodreads runs.

| Dataset | CSV in `Data/` | Epochs | Samples (train / val / test) | Sample files | Tar |
|---|---|---|---|---|---|
| Anime | `MAL_HSAL_Final.csv` (155,471 users, 5,961 items) | 10 | 200,995 / 36,457 / 54,994 | 292,446 | 8.9 GB |
| ML-1M | `MovieLens_all_v2.csv` (3,706 movies) | 5 | not recorded | 994,169 | 33 GB |

Settings for every run: seed 0, batch 256, k_hop 2, lr 0.001, l2 0.0001, 3 layers, hidden size 50, item/user
length 50, CCR `faculty` scavenger with `--requeue` and `--resume`.

## Files

| File | What it is |
|---|---|
| `anime_ml1m_jobs.sbatch` | One job file for every step; `MODE` picks the step |
| `anime_ml1m_all_epochs.csv` | Recall and NDCG at 5/10/20/50/100 for every epoch of all 8 runs |
| `anime_ml1m_results.md` | This summary |

## How the runs were made

Copy `anime_ml1m_jobs.sbatch` into `/projects/academic/haimonti/snirwade/HSAL-Chunking`, then for each dataset:

```bash
sbatch --export=ALL,DATASET=MAL_HSAL_Final,MODE=hsal_chunked  --job-name=HSALchunk_Anime anime_ml1m_jobs.sbatch
sbatch --export=ALL,DATASET=MAL_HSAL_Final,MODE=lrd_chunked   --job-name=LRDchunk_Anime  anime_ml1m_jobs.sbatch
sbatch --export=ALL,DATASET=MAL_HSAL_Final,MODE=verify        --job-name=verify_Anime    anime_ml1m_jobs.sbatch
sbatch --export=ALL,DATASET=MAL_HSAL_Final,MODE=lrd_original  --job-name=LRDorig_Anime   anime_ml1m_jobs.sbatch

sbatch --export=ALL,DATASET=MovieLens_all_v2,MODE=hsal_chunked,EPOCHS=5  --job-name=HSALchunk_ML1M anime_ml1m_jobs.sbatch
sbatch --export=ALL,DATASET=MovieLens_all_v2,MODE=lrd_chunked,EPOCHS=5   --job-name=LRDchunk_ML1M  anime_ml1m_jobs.sbatch
sbatch --export=ALL,DATASET=MovieLens_all_v2,MODE=hsal_original,EPOCHS=5 --job-name=HSALorig_ML1M  anime_ml1m_jobs.sbatch
sbatch --export=ALL,DATASET=MovieLens_all_v2,MODE=lrd_original,EPOCHS=5  --job-name=LRDorig_ML1M   anime_ml1m_jobs.sbatch
```

- `lrd_chunked` waits for the `hsal_chunked` preprocessing; `lrd_original` needs the `hsal_original` preprocessing.
- Anime original HSAL was the earlier run `mal_hsal_r0` (job 24924331, seed 0), so `hsal_original` was not rerun for Anime.
- The ML-1M item embeddings (`Data/MovieLens_all_v2{,_chunked}_llm_embeddings.pt`, 3,706 x 384, all-MiniLM-L6-v2,
  text "Title: X. Genres: a, b.") were made once with `MODE=ml1m_embeddings bash anime_ml1m_jobs.sbatch` on a
  machine with sentence-transformers, because the CCR venv does not have it.
- Samples are built on the node's local disk and kept as one tar per dataset in `archives/`, instead of hundreds of
  thousands of small files in `/projects`.
- The job file combines the separate scripts used for these runs (same steps and settings); the combined file was
  syntax- and dispatch-checked but has not itself been run end to end.

## Verification (Anime)

Fresh chunked preprocessing compared with the original outputs (job 25230924): **PASSED**.

- users 155,471 and items 5,961: same sets
- graph edges 500,000: identical (user, item, time) edges
- all 292,446 samples: same target and identical subgraph edges
- negatives: identical unread-item sets for 500/500 sampled users
- series table: 5,461 rows, identical

## Results (Recall@10 / NDCG@10, best epoch)

| Dataset | Model | Original | Chunked | Difference |
|---|---|---|---|---|
| Anime | HSAL | 0.6222 / 0.3127 | 0.6116 / 0.3083 | -0.0106 / -0.0044 |
| Anime | HSAL+LRD | 0.6338 / 0.3154 | 0.6621 / 0.3347 | +0.0283 / +0.0193 |
| ML-1M | HSAL | 0.6623 / 0.3859 | 0.6588 / 0.3849 | -0.0035 / -0.0010 |
| ML-1M | HSAL+LRD | 0.7075 / 0.4333 | 0.7070 / 0.4331 | -0.0005 / -0.0002 |

HSAL+LRD chunked vs HSAL chunked: +0.0505 / +0.0264 on Anime, +0.0482 / +0.0482 on ML-1M.

The training script reports the best value so far at each epoch, so the last epoch in the CSV is the best epoch.
Each run uses one seed, so differences this small are not tested for run-to-run variation.

## Jobs and run time

| Dataset | Run | Slurm job | Wall time (incl. preprocessing) |
|---|---|---|---|
| Anime | HSAL original | 24924331 | earlier run |
| Anime | HSAL chunked | 25230222 | 6 h 26 min |
| Anime | HSAL+LRD original | 25230925 | 2 h 19 min |
| Anime | HSAL+LRD chunked | 25230880 | 1 h 13 min |
| Anime | verification | 25230924 | 12 min |
| ML-1M | HSAL original | 25231096 | 13 h 44 min |
| ML-1M | HSAL chunked | 25231071 | 23 h 42 min |
| ML-1M | HSAL+LRD original | 25231803 | 3 h 33 min |
| ML-1M | HSAL+LRD chunked | 25231095 | 3 h 22 min |

Preprocessing took about 6 min for Anime chunked and about 16 min for each ML-1M pipeline. ML-1M HSAL chunked trained at
about 4.5 s per batch against 2.5 s for the original, but the two ran on different, shared nodes, so the gap is not
attributed to chunking.
