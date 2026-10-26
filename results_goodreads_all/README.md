# HSAL and HSAL+LRD on four Goodreads genres (chunked graph construction)

Snapshot taken 26 Oct 2026, 09:40. All runs: 10 epochs, seed 0, batch 256, k_hop 2, 8 user-disjoint chunks,
CCR scavenger, `--resume`. Values are the best so far up to that epoch (that is how `*_results.txt` records them).

**Still running at this snapshot:** HSAL chunked on Romance (job 25157774, epoch 7 of 10) and on Mystery
(job 25157775, epoch 10 of 10). Their rows stop at epochs 6 and 9; rerun the collection when they finish.

## Datasets

| Dataset | Rows | Users | Books | Users with >= 3 books |
|---|---|---|---|---|
| Fantasy & Paranormal | 599,097 | 192,835 | 29,428 | 57,463 |
| Young Adult | 272,339 | 116,146 | 8,518 | 26,392 |
| Romance | 416,985 | 110,439 | 37,811 | 35,340 |
| Mystery, Thriller & Crime | 399,905 | 148,638 | 15,292 | 35,198 |

## Latest result per run (Recall@10 / NDCG@10)

| Dataset | HSAL original | HSAL chunked | HSAL+LRD original | HSAL+LRD chunked |
|---|---|---|---|---|
| Fantasy | 0.5391 / 0.3028 | 0.5422 / 0.3044 | 0.5437 / 0.3081 | 0.5466 / 0.3110 |
| Young Adult | – | 0.3826 / 0.1956 | – | 0.3939 / 0.2035 |
| Romance | – | 0.4688 / 0.2583 (epoch 6, running) | – | 0.4964 / 0.2787 |
| Mystery | – | 0.4536 / 0.2482 (epoch 9, running) | – | 0.4808 / 0.2659 |

## Same-epoch comparison, HSAL+LRD chunked minus HSAL chunked

| Dataset | Epoch | Recall@10 | NDCG@10 |
|---|---|---|---|
| Fantasy | 10 | +0.0044 | +0.0066 |
| Young Adult | 10 | +0.0113 | +0.0079 |
| Romance | 6 | -0.0033 | -0.0021 |
| Mystery | 9 | +0.0183 | +0.0115 |

## Jobs and GPUs

| Run | Job | GPU |
|---|---|---|
| Fantasy HSAL original / chunked | 25107241 / 25107242 | L40S (original was requeued; last attempt on an H100 node) / L40S |
| Fantasy HSAL+LRD original / chunked | 25153950 / 25153951 | A100 80GB / L40S |
| Young Adult HSAL / HSAL+LRD chunked | 25157773 / 25157776 | A100 PCIe 40GB / A100 PCIe 40GB |
| Romance HSAL / HSAL+LRD chunked | 25157774 / 25157777 | A100 SXM4 80GB / A100 SXM4 80GB |
| Mystery HSAL / HSAL+LRD chunked | 25157775 / 25157778 | A100 SXM4 80GB / A100 SXM4 80GB |

## Notes

- Chunked vs original was checked on Fantasy only: identical samples, final results within 0.003.
- HSAL+LRD uses the bug-fixed `HSAL_for_lrd_fixed.py` and title + description embeddings
  (all-MiniLM-L6-v2); text was found for every book in all four genres.
- One seed per run, so differences under about 0.005 should not be read as real.

Files: `all_epochs.csv` (every epoch of every run), `recall10_chart.png`, `ccr_output.txt` (job states, GPUs,
raw rows). The same folder is on CCR at `HSAL-Chunking/comparison_goodreads_all/` and
`/projects/academic/haimonti/snirwade/comparison_goodreads_all/`.
