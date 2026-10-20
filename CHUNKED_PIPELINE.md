# Chunked HSAL preprocessing

Chunk-by-chunk copies of the HSAL preprocessing scripts. The original files are unchanged.

| New file | Copy of | What changed |
|---|---|---|
| `split_hsal_csv_by_user.py` | (new) | Splits one HSAL CSV into N user-disjoint chunk CSVs |
| `new_data1_chunked.py` | `new_data1.py` | Global user/book ID maps; graph built chunk by chunk (`add-chunk`) or in one shot (`build-graph`); samples generated per chunk of users (`samples --chunks k`) |
| `sinusoidal_chunked.py` | `sinusoidal.py` | Reads all chunks, uses the global book IDs |
| `generate_neg_chunked.py` | `generate_neg.py` | Reads all chunks, uses the global IDs; stores book IDs as int32 (same values, half the size) |
| `new_main_BPR_chunked.py` | `new_main_BPR.py` | User/book counts come from the ID maps instead of the single CSV |
| `run_fantasy_baseline.sbatch`, `fantasy_baseline_steps.sh` | `run_job.sbatch` | CCR job: original pipeline on the single Fantasy CSV |
| `run_fantasy_chunked.sbatch`, `fantasy_chunked_steps.sh` | `run_job.sbatch` | CCR job: same code and settings, data added in 8 chunks |
| `fantasy_common.sh` | (new) | Settings shared by both Fantasy jobs |
| `verify_chunked_vs_original.py` | (new) | Compares chunked outputs with the original pipeline |

## Steps

```bash
python split_hsal_csv_by_user.py --data Goodreads_young_adult_HSAL_100k --n 8
for k in 1 2 3 4 5 6 7 8; do
  python new_data1_chunked.py add-chunk --data Goodreads_young_adult_HSAL_100k_chunked \
      --chunk Data/Goodreads_young_adult_HSAL_100k_chunks/Goodreads_young_adult_HSAL_100k_chunk_$k.csv
done
python new_data1_chunked.py finalize --data Goodreads_young_adult_HSAL_100k_chunked
python sinusoidal_chunked.py --data Goodreads_young_adult_HSAL_100k_chunked
python generate_neg_chunked.py --data Goodreads_young_adult_HSAL_100k_chunked
for k in 1 2 3 4 5 6 7 8; do   # can run as separate jobs, after ALL chunks are added
  python new_data1_chunked.py samples --data Goodreads_young_adult_HSAL_100k_chunked --chunks $k
done
python new_main_BPR_chunked.py --data Goodreads_young_adult_HSAL_100k_chunked --model_record
```

Chunk CSVs must have the HSAL columns `user_id,item_id,time,item_series_id`. The raw Goodreads
shards from the Colab partition test do not have them yet; they first need the series/edition
preprocessing from the Goodreads notebook.

## Rules

- Add every chunk before generating samples. A user's subgraph pulls in other users who read the
  same books, and those users can be in any chunk.
- IDs are assigned once: users in order of first appearance, books the first time they are seen.
  `Data/<name>_chunked/users.csv` and `books.csv` map raw IDs to model IDs.
- Adding the same chunk twice is skipped; a chunk containing users from an earlier chunk is rejected.
- `--data` must end with `_chunked`, so the chunked scripts can never overwrite the original
  `Data/<name>_graph`, `<name>_neg`, series table or `Newdata/<name>_...` outputs.
- `samples`, `sinusoidal_chunked.py` and `generate_neg_chunked.py` refuse to run until all
  `--n_chunks` chunks (default 8) are added; `samples` also refuses a graph that is missing chunks.
- `finalize` must run after the last chunk and before samples, series table and negatives (they refuse
  otherwise). It refuses to run if any of those outputs already exist, since they would use old IDs.
  It only renumbers: the (user, book, time) edges are the same before and after.
- Sequels across chunks: a book read only in chunk 3 and its sequel read only in chunk 5 still get the
  same `series_id` in the series table (845 such pairs in the test slice), because the table is built
  from all chunks.

## Check against the original (12,000-user slice of Young Adult 100k)

Identical: users, books, all 36,850 graph edges (user, book, time), sample counts
(14,766 train / 2,652 val / 3,824 test), every sample's target book, negatives, series table.

Before `finalize` existed, about 15% of subgraphs differed: users were numbered chunk by chunk, and
when more than 50 neighbour users are found `generate_user` keeps the 50 with the largest ID numbers
(`np.setdiff1d(...)[-user_max_length:]`). `finalize` renumbers users and books in original-file order
(`orig_row`, written by the split script), which removes the difference without changing the rule.

## Fantasy full run on CCR (Goodreads_Fantasy_HSAL_Final, 599,097 rows)

Both jobs: scavenger, NVIDIA L40S, 10 epochs, seed 0, batch 256, k_hop 2, `--model_record --resume`,
run from `/vscratch/grp-haimonti/snirwade/HSAL-Chunking` (the container needs
`-B /vscratch/grp-haimonti/snirwade,/projects/academic/haimonti` because `venv` is a link into `/projects`).

| | Baseline (original) | Chunked (8 chunks) |
|---|---|---|
| Job | 25107241 | 25107242 |
| Samples train / val / test | 254,807 / 38,998 / 57,463 | same |
| Preprocessing | ~7 h | 8 min |
| Training | ~1 h per epoch (pre-empted once, resumed) | ~1 h per epoch |
| Recall@10 / NDCG@10 after 10 epochs (best so far) | 0.5391 / 0.3028 | 0.5422 / 0.3044 |

Per-epoch numbers: `results_chunking_fantasy/comparison.csv`. Across epochs 1-9 the two runs differ by at
most 0.0008; the samples are identical, so remaining differences come from training randomness.
