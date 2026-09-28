"""
Prequel matching driver. Relation-specific: direction='before', file paths,
output naming. Matching mechanics live in matching_core.py.

Performance note: cosine-candidate query texts are batch-encoded in a single
model.encode() call instead of one-at-a-time in a loop. This does not change
the matching methodology (same embeddings, same threshold, same reranking) —
it only avoids the huge per-call overhead of unbatched CPU inference, which
was the entire cause of the full-corpus run being slow (400-book scale had
few enough cosine candidates for the unbatched loop to finish in under a
minute; full-corpus scale has ~10-13x more, so the same unbatched loop
scaled linearly instead of amortizing).
"""

import argparse
import csv
import numpy as np
import pandas as pd

from matching_core import (
    EXPERIMENTAL_THRESHOLD,
    normalize_title,
    cosine_sim_matrix,
    make_coherence_fn,
    resolve_exact_match,
    rerank_by_series_position,
)

DIRECTION = "before"  # prequel: candidate must come BEFORE the central book

SAVE_DIR = './'
LLM_DIR = './prequel_extraction/'

CORPUS_FILES = {
    "A": {
        "meta": SAVE_DIR + 'Goodreads_young_adult_item_meta_with_title.csv',
        "feat": SAVE_DIR + 'Goodreads_young_adult.feat.npy',
        "idx": SAVE_DIR + 'Goodreads_young_adult.item2index',
    },
    "B": {
        "meta": SAVE_DIR + 'Goodreads_young_adult_author.meta.csv',
        "feat": SAVE_DIR + 'Goodreads_young_adult_author.feat.npy',
        "idx": SAVE_DIR + 'Goodreads_young_adult_author.item2index',
    },
}

LLM_FILES_BY_VERSION = {
    "A": {
        'llama3.1_8b': LLM_DIR + 'prequels_sample400_llama3.1_8b_verified.csv',
        'mistral': LLM_DIR + 'prequels_sample400_mistral_verified.csv',
    },
    "B": {
        'llama3.1_8b': LLM_DIR + 'prequels_sample400_vB_llama3.1_8b_verified.csv',
        'mistral': LLM_DIR + 'prequels_sample400_vB_mistral_verified.csv',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["A", "B"], default="A")
    parser.add_argument("--test-file", default=None,
                         help="Override: point at a single extraction output CSV "
                              "(e.g. the full 8,108-book corpus run) instead of the "
                              "default 400-book LLM_FILES dict.")
    parser.add_argument("--test-label", default="test_run",
                         help="Label used in the output filename when --test-file is set.")
    parser.add_argument("--batch-size", type=int, default=128,
                         help="Batch size for encoding cosine-candidate query texts.")
    args = parser.parse_args()

    META_FILE = CORPUS_FILES[args.version]["meta"]
    FEAT_FILE = CORPUS_FILES[args.version]["feat"]
    IDX_FILE = CORPUS_FILES[args.version]["idx"]

    llm_files = {args.test_label: args.test_file} if args.test_file else LLM_FILES_BY_VERSION[args.version]

    print("Loading corpus...")
    meta = pd.read_csv(META_FILE)
    meta['norm_title'] = meta['title'].apply(normalize_title)

    title_to_ids = {}
    for _, row in meta.iterrows():
        title_to_ids.setdefault(row['norm_title'], []).append(row['item_id'])

    embeddings = np.load(FEAT_FILE)
    item2index = {}
    with open(IDX_FILE, 'r') as f:
        for line in f:
            item_id, idx = line.strip().split('\t')
            item2index[int(item_id)] = int(idx)
    index2item = {v: k for k, v in item2index.items()}
    id_to_title = dict(zip(meta['item_id'], meta['title']))
    coherence = make_coherence_fn(embeddings, item2index)

    print(f"  Corpus: {len(meta):,} items, embeddings shape {embeddings.shape}")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess
        subprocess.call(['pip', 'install', 'sentence-transformers', '-q'])
        from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    for model_name, llm_file in llm_files.items():
        print(f"\n=== {model_name} ===")
        df = pd.read_csv(llm_file)
        df['has_prequel'] = df['has_prequel'].astype(str).str.lower() == 'true'
        true_rows = df[df['has_prequel'] & df['match_title'].notna() & (df['match_title'] != '')]
        print(f"  Books with a match to check: {len(true_rows)}")

        results = []
        cosine_pending = []  # (central_id, central_title, llm_match_title, query_text)

        # Pass 1: resolve exact matches immediately, queue everything else for batch encoding
        for _, row in true_rows.iterrows():
            central_id = int(row['item_id'])
            m_title = str(row['match_title'])
            m_desc = str(row.get('match_description', '') or '')
            norm_m = normalize_title(m_title)

            candidates = [i for i in title_to_ids.get(norm_m, []) if i != central_id]

            if candidates:
                matched_id, series_status, _ = resolve_exact_match(
                    central_id, row['title'], candidates, id_to_title
                )
                if matched_id is not None:
                    coh = coherence(central_id, matched_id)
                    flag = (series_status != 'match' and coh is not None and coh < EXPERIMENTAL_THRESHOLD)
                    results.append({
                        'item_id': central_id, 'central_title': row['title'], 'llm_match_title': m_title,
                        'match_type': 'exact', 'matched_item_id': matched_id,
                        'matched_title': id_to_title.get(matched_id, ''),
                        'top1_score': 1.0, 'top1_item_id': matched_id,
                        'top3_matches': '', 'top5_matches': '', 'above_experimental_threshold': True,
                        'central_matched_coherence': round(coh, 4) if coh is not None else '',
                        'low_coherence_flag': flag, 'series_match': series_status,
                    })
                else:
                    results.append({
                        'item_id': central_id, 'central_title': row['title'], 'llm_match_title': m_title,
                        'match_type': 'exact_rejected_series_mismatch', 'matched_item_id': '', 'matched_title': '',
                        'top1_score': 1.0, 'top1_item_id': candidates[0],
                        'top3_matches': f"rejected: {id_to_title.get(candidates[0], '')}",
                        'top5_matches': '', 'above_experimental_threshold': False,
                        'central_matched_coherence': '', 'low_coherence_flag': True, 'series_match': 'mismatch',
                    })
                continue

            query_text = f"{m_title} {m_desc}".strip()
            cosine_pending.append((central_id, row['title'], m_title, query_text))

        # Pass 2: batch-encode every queued query text in one call, then compute cosine/coherence per row
        if cosine_pending:
            print(f"  Batch-encoding {len(cosine_pending):,} cosine-candidate queries "
                  f"(batch_size={args.batch_size})...")
            query_texts = [qt for (_, _, _, qt) in cosine_pending]
            query_vecs = model.encode(
                query_texts, convert_to_numpy=True, batch_size=args.batch_size, show_progress_bar=True
            )

            for (central_id, central_title, m_title, _), query_vec in zip(cosine_pending, query_vecs):
                sims = cosine_sim_matrix(query_vec, embeddings)
                central_idx = item2index.get(central_id)
                if central_idx is not None:
                    sims[central_idx] = -1.0

                top5_idx = np.argsort(-sims)[:5]
                top5 = [(index2item[i], round(float(sims[i]), 4)) for i in top5_idx]

                top1_id, top1_score, rerank_used = rerank_by_series_position(
                    central_title, top5, id_to_title, direction=DIRECTION
                )

                coh = coherence(central_id, top1_id) if top1_score >= EXPERIMENTAL_THRESHOLD else None
                results.append({
                    'item_id': central_id, 'central_title': central_title, 'llm_match_title': m_title,
                    'match_type': 'cosine_reranked_by_series_position' if rerank_used else 'cosine',
                    'matched_item_id': top1_id if top1_score >= EXPERIMENTAL_THRESHOLD else '',
                    'matched_title': id_to_title.get(top1_id, '') if top1_score >= EXPERIMENTAL_THRESHOLD else '',
                    'top1_score': top1_score, 'top1_item_id': top1_id,
                    'top3_matches': '; '.join(f"{id_to_title.get(i,'')} ({s})" for i, s in top5[:3]),
                    'top5_matches': '; '.join(f"{id_to_title.get(i,'')} ({s})" for i, s in top5),
                    'above_experimental_threshold': top1_score >= EXPERIMENTAL_THRESHOLD,
                    'central_matched_coherence': round(coh, 4) if coh is not None else '',
                    'low_coherence_flag': (coh is not None and coh < EXPERIMENTAL_THRESHOLD),
                    'series_match': '',
                })

        out_df = pd.DataFrame(results)
        out_suffix = '' if args.version == 'A' else f'_v{args.version}'
        base_label = 'full_corpus' if args.test_file else 'sample400'
        out_path = SAVE_DIR + f'matches_{base_label}_prequel_{model_name}{out_suffix}.csv'
        out_df.to_csv(out_path, index=False)

        n_exact = (out_df['match_type'] == 'exact').sum()
        n_exact_rejected = (out_df['match_type'] == 'exact_rejected_series_mismatch').sum()
        is_cosine_family = out_df['match_type'].isin(['cosine', 'cosine_reranked_by_series_position'])
        n_cosine_above = (is_cosine_family & out_df['above_experimental_threshold']).sum()
        n_flagged = out_df['low_coherence_flag'].sum()
        print(f"  Exact accepted: {n_exact}   Exact rejected (series mismatch): {n_exact_rejected}")
        print(f"  Cosine matches >= {EXPERIMENTAL_THRESHOLD}: {n_cosine_above}   Flagged: {n_flagged}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
