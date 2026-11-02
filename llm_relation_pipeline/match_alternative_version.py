"""
Alternative-version matching driver. Unlike Prequel/Sequel, this relation has
NO positional meaning — matching_core.py's series-position rerank does not
apply here, only exact match + cosine + coherence, and only for medium='book'
claims. Movie/TV/anime claims are recorded as identified-but-unmatched, since
no target embedding corpus exists for those datasets yet.
"""

import argparse
import numpy as np
import pandas as pd

from matching_core import (
    EXPERIMENTAL_THRESHOLD,
    normalize_title,
    cosine_sim_matrix,
    make_coherence_fn,
    resolve_exact_match,
)

SAVE_DIR = './'
LLM_DIR = './alternative_version_extraction/'

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
        'llama3.1_8b': LLM_DIR + 'altver_eval_llama3.1_8b_verified.csv',
        'mistral': LLM_DIR + 'altver_eval_mistral_verified.csv',
    },
    "B": {
        'llama3.1_8b': LLM_DIR + 'altver_eval_vB_llama3.1_8b_verified.csv',
        'mistral': LLM_DIR + 'altver_eval_vB_mistral_verified.csv',
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["A", "B"], default="A")
    parser.add_argument("--test-file", default=None)
    parser.add_argument("--test-label", default="test_run")
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
        df['has_alternative_version'] = df['has_alternative_version'].astype(str).str.lower() == 'true'
        true_rows = df[df['has_alternative_version'] & df['match_title'].notna() & (df['match_title'] != '')]
        print(f"  Books with a claimed alternative version: {len(true_rows)}")

        results = []
        for _, row in true_rows.iterrows():
            central_id = int(row['item_id'])
            m_title = str(row['match_title'])
            m_desc = str(row.get('match_description', '') or '')
            medium = str(row.get('match_medium', '') or '').strip().lower()

            if medium and medium != 'book':
                results.append({
                    'item_id': central_id, 'central_title': row['title'], 'llm_match_title': m_title,
                    'match_medium': medium, 'match_type': 'no_target_corpus',
                    'matched_item_id': '', 'matched_title': '',
                    'top1_score': '', 'top1_item_id': '', 'top3_matches': '', 'top5_matches': '',
                    'above_experimental_threshold': False, 'central_matched_coherence': '',
                    'low_coherence_flag': False, 'series_match': '',
                })
                continue

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
                        'match_medium': 'book', 'match_type': 'exact', 'matched_item_id': matched_id,
                        'matched_title': id_to_title.get(matched_id, ''),
                        'top1_score': 1.0, 'top1_item_id': matched_id,
                        'top3_matches': '', 'top5_matches': '', 'above_experimental_threshold': True,
                        'central_matched_coherence': round(coh, 4) if coh is not None else '',
                        'low_coherence_flag': flag, 'series_match': series_status,
                    })
                else:
                    results.append({
                        'item_id': central_id, 'central_title': row['title'], 'llm_match_title': m_title,
                        'match_medium': 'book', 'match_type': 'exact_rejected_series_mismatch',
                        'matched_item_id': '', 'matched_title': '',
                        'top1_score': 1.0, 'top1_item_id': candidates[0],
                        'top3_matches': f"rejected: {id_to_title.get(candidates[0], '')}",
                        'top5_matches': '', 'above_experimental_threshold': False,
                        'central_matched_coherence': '', 'low_coherence_flag': True, 'series_match': 'mismatch',
                    })
                continue

            query_text = f"{m_title} {m_desc}".strip()
            query_vec = model.encode([query_text], convert_to_numpy=True)[0]
            sims = cosine_sim_matrix(query_vec, embeddings)
            central_idx = item2index.get(central_id)
            if central_idx is not None:
                sims[central_idx] = -1.0

            top5_idx = np.argsort(-sims)[:5]
            top5 = [(index2item[i], round(float(sims[i]), 4)) for i in top5_idx]
            top1_id, top1_score = top5[0]

            coh = coherence(central_id, top1_id) if top1_score >= EXPERIMENTAL_THRESHOLD else None
            results.append({
                'item_id': central_id, 'central_title': row['title'], 'llm_match_title': m_title,
                'match_medium': 'book', 'match_type': 'cosine',
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
        out_path = SAVE_DIR + f'matches_altver_{model_name}{out_suffix}.csv'
        out_df.to_csv(out_path, index=False)

        n_no_corpus = (out_df['match_type'] == 'no_target_corpus').sum()
        n_exact = (out_df['match_type'] == 'exact').sum()
        n_exact_rejected = (out_df['match_type'] == 'exact_rejected_series_mismatch').sum()
        n_cosine_above = ((out_df['match_type'] == 'cosine') & out_df['above_experimental_threshold']).sum()
        n_flagged = out_df['low_coherence_flag'].sum()
        print(f"  Non-book medium (movie/tv/anime, no target corpus yet): {n_no_corpus}")
        print(f"  Exact accepted: {n_exact}   Exact rejected (series mismatch): {n_exact_rejected}")
        print(f"  Cosine matches >= {EXPERIMENTAL_THRESHOLD}: {n_cosine_above}   Flagged: {n_flagged}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()