"""
Stage 7: match LLM-identified prequels to the existing HSAL graph.

For each book the LLM said has a prequel, try (in order):
  1. Exact match: normalize both the LLM's prequel_title and every corpus
     title (strip series-annotation suffix, lowercase, strip punctuation),
     compare. If found, no embedding needed for this book.
  2. Cosine similarity fallback: embed prequel_title + prequel_description
     with the same model used for the corpus (all-MiniLM-L6-v2), compare
     against every corpus embedding, return top-5.

similarity threshold (0.5) is EXPERIMENTAL, not a fixed cutoff — the output
keeps the full top-5 ranked list so the actual threshold can be tuned
after looking at real numbers, not assumed in advance.
"""

import ast
import csv
import re
import numpy as np
import pandas as pd

SAVE_DIR = './'
LLM_DIR = './prequel_extraction/'

META_FILE = SAVE_DIR + 'Goodreads_young_adult_item_meta_with_title.csv'
FEAT_FILE = SAVE_DIR + 'Goodreads_young_adult.feat.npy'
IDX_FILE = SAVE_DIR + 'Goodreads_young_adult.item2index'

LLM_FILES = {
    'llama3.1_8b': LLM_DIR + 'prequels_sample400_llama3.1_8b_verified.csv',
    'mistral': LLM_DIR + 'prequels_sample400_mistral_verified.csv',
}

EXPERIMENTAL_THRESHOLD = 0.5  # not a final cutoff — reported alongside full top-5 for later tuning

SERIES_SUFFIX_RE = re.compile(r'\s*\([^)]*\)\s*$')
PUNCT_RE = re.compile(r'[^a-z0-9\s]')


def normalize_title(title):
    t = title.strip()
    t = SERIES_SUFFIX_RE.sub('', t)  # strip trailing "(Series, #N)"
    t = t.lower()
    t = PUNCT_RE.sub('', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


SERIES_POS_RE = re.compile(r'\(([^,()]+),\s*#(\d+)\)\s*$')


def parse_series_position(title):
    """Extract (normalized_series_name, position) only for a bare integer
    position like '#1', '#2', '#37' — NOT '#1.5', '#4.4', or '#0', which are
    novellas/interstitials/prequel-markers, not reliable main-line ordering."""
    m = SERIES_POS_RE.search(title.strip())
    if not m:
        return None, None
    series_raw, pos_str = m.group(1), m.group(2)
    pos = int(pos_str)
    if pos == 0:
        return None, None
    series_norm = re.sub(r'\s+', ' ', PUNCT_RE.sub('', series_raw.lower())).strip()
    return series_norm, pos


SERIES_NAME_RE = re.compile(r'\(([^()]+)\)\s*$')


def parse_series_name(title):
    """Extract just the series name from a trailing parenthetical, regardless
    of whether it carries a bare-integer position, a decimal, or no number at
    all (e.g. '(Pure, #1)', '(The Fine Art of Pretending, #1)', '(Cherub, #7)').
    Returns None if the title has no parenthetical to parse — this is NOT the
    same as a mismatch; it means series identity can't be checked either way."""
    m = SERIES_NAME_RE.search(title.strip())
    if not m:
        return None
    inner = re.sub(r',?\s*#[\w.]+\s*$', '', m.group(1)).strip()
    if not inner:
        return None
    return normalize_title(inner)


def cosine_sim_matrix(query_vec, corpus_matrix):
    q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    c = corpus_matrix / (np.linalg.norm(corpus_matrix, axis=1, keepdims=True) + 1e-10)
    return c @ q


def main():
    print("Loading corpus...")
    meta = pd.read_csv(META_FILE)
    meta['norm_title'] = meta['title'].apply(normalize_title)

    # normalized title -> list of item_ids (a title can be non-unique)
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

    print(f"  Corpus: {len(meta):,} items, embeddings shape {embeddings.shape}")

    def coherence(id_a, id_b):
        """Cosine similarity between two corpus books' own embeddings —
        used to sanity-check that a matched book is topically plausible
        as a prequel, independent of whether the LLM's title string matched."""
        idx_a, idx_b = item2index.get(id_a), item2index.get(id_b)
        if idx_a is None or idx_b is None:
            return None
        va, vb = embeddings[idx_a], embeddings[idx_b]
        return float(
            (va @ vb) / ((np.linalg.norm(va) + 1e-10) * (np.linalg.norm(vb) + 1e-10))
        )

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        import subprocess
        subprocess.call(['pip', 'install', 'sentence-transformers', '-q'])
        from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('all-MiniLM-L6-v2')

    for model_name, llm_file in LLM_FILES.items():
        print(f"\n=== {model_name} ===")
        df = pd.read_csv(llm_file)
        df['has_prequel'] = df['has_prequel'].astype(str).str.lower() == 'true'
        true_rows = df[df['has_prequel'] & df['prequel_title'].notna() & (df['prequel_title'] != '')]
        print(f"  Books with has_prequel=True and a title: {len(true_rows)}")

        results = []
        for _, row in true_rows.iterrows():
            central_id = int(row['item_id'])
            pq_title = str(row['prequel_title'])
            pq_desc = str(row.get('prequel_description', '') or '')
            norm_pq = normalize_title(pq_title)

            candidates = [i for i in title_to_ids.get(norm_pq, []) if i != central_id]

            if candidates:
                central_series_name = parse_series_name(row['title'])
                cand_info = []
                for cid in candidates:
                    cand_series_name = parse_series_name(id_to_title.get(cid, ''))
                    if central_series_name is None or cand_series_name is None:
                        status = 'unknown'  # can't check either way — not a mismatch
                    elif central_series_name == cand_series_name:
                        status = 'match'
                    else:
                        status = 'mismatch'
                    cand_info.append((cid, status))

                # Prefer a confirmed same-series candidate; else an unverifiable
                # one; only reject outright if EVERY candidate is a confirmed
                # different series (a coincidental title collision).
                chosen = next((c for c in cand_info if c[1] == 'match'), None)
                if chosen is None:
                    chosen = next((c for c in cand_info if c[1] == 'unknown'), None)

                if chosen is not None:
                    matched_id, series_status = chosen
                    coh = coherence(central_id, matched_id)
                    results.append({
                        'item_id': central_id,
                        'central_title': row['title'],
                        'llm_prequel_title': pq_title,
                        'match_type': 'exact',
                        'matched_item_id': matched_id,
                        'matched_title': id_to_title.get(matched_id, ''),
                        'top1_score': 1.0,
                        'top1_item_id': matched_id,
                        'top3_matches': '',
                        'top5_matches': '',
                        'above_experimental_threshold': True,
                        'central_matched_coherence': round(coh, 4) if coh is not None else '',
                        'low_coherence_flag': (
                            series_status != 'match'
                            and coh is not None and coh < EXPERIMENTAL_THRESHOLD
                        ),
                        'series_match': series_status,
                    })
                else:
                    # every title-string candidate is a confirmed different series —
                    # a coincidental collision, not a real prequel relationship
                    results.append({
                        'item_id': central_id,
                        'central_title': row['title'],
                        'llm_prequel_title': pq_title,
                        'match_type': 'exact_rejected_series_mismatch',
                        'matched_item_id': '',
                        'matched_title': '',
                        'top1_score': 1.0,
                        'top1_item_id': cand_info[0][0],
                        'top3_matches': f"rejected: {id_to_title.get(cand_info[0][0], '')}",
                        'top5_matches': '',
                        'above_experimental_threshold': False,
                        'central_matched_coherence': '',
                        'low_coherence_flag': True,
                        'series_match': 'mismatch',
                    })
                continue

            # Fallback: cosine similarity
            query_text = f"{pq_title} {pq_desc}".strip()
            query_vec = model.encode([query_text], convert_to_numpy=True)[0]
            sims = cosine_sim_matrix(query_vec, embeddings)

            # exclude the central book's own row from candidacy
            central_idx = item2index.get(central_id)
            if central_idx is not None:
                sims[central_idx] = -1.0

            top5_idx = np.argsort(-sims)[:5]
            top5 = [(index2item[i], round(float(sims[i]), 4)) for i in top5_idx]

            # Series-position rerank: among the top-5, prefer a candidate in the
            # SAME series with a strictly earlier, reliable (bare-integer) position
            # over whichever candidate merely scored highest on raw similarity.
            central_series, central_pos = parse_series_position(row['title'])
            reranked_id, reranked_score, rerank_used = None, None, False
            if central_series and central_pos:
                same_series_earlier = []
                for cand_id, cand_score in top5:
                    cand_title = id_to_title.get(cand_id, '')
                    cand_series, cand_pos = parse_series_position(cand_title)
                    if cand_series == central_series and cand_pos is not None \
                            and cand_pos < central_pos and (central_pos - cand_pos) <= 3:
                        same_series_earlier.append((cand_id, cand_score, cand_pos))
                if same_series_earlier:
                    # prefer the immediate predecessor (largest qualifying position),
                    # not just whichever one happened to score highest
                    same_series_earlier.sort(key=lambda x: -x[2])
                    reranked_id, reranked_score, _ = same_series_earlier[0]
                    rerank_used = True

            if rerank_used:
                top1_id, top1_score = reranked_id, reranked_score
            else:
                top1_id, top1_score = top5[0]
            matched_for_coherence = top1_id if top1_score >= EXPERIMENTAL_THRESHOLD else None
            coh = coherence(central_id, matched_for_coherence) if matched_for_coherence else None
            results.append({
                'item_id': central_id,
                'central_title': row['title'],
                'llm_prequel_title': pq_title,
                'match_type': 'cosine_reranked_by_series_position' if rerank_used else 'cosine',
                'matched_item_id': top1_id if top1_score >= EXPERIMENTAL_THRESHOLD else '',
                'matched_title': id_to_title.get(top1_id, '') if top1_score >= EXPERIMENTAL_THRESHOLD else '',
                'top1_score': top1_score,
                'top1_item_id': top1_id,
                'top3_matches': '; '.join(f"{id_to_title.get(i,'')} ({s})" for i, s in top5[:3]),
                'top5_matches': '; '.join(f"{id_to_title.get(i,'')} ({s})" for i, s in top5),
                'above_experimental_threshold': top1_score >= EXPERIMENTAL_THRESHOLD,
                'central_matched_coherence': round(coh, 4) if coh is not None else '',
                'low_coherence_flag': (coh is not None and coh < EXPERIMENTAL_THRESHOLD),
                'series_match': '',  # not applicable to cosine matches — series check is exact-match only for now
            })

        out_df = pd.DataFrame(results)
        out_path = SAVE_DIR + f'prequel_matches_sample400_{model_name}.csv'
        out_df.to_csv(out_path, index=False)

        n_exact = (out_df['match_type'] == 'exact').sum()
        n_exact_rejected = (out_df['match_type'] == 'exact_rejected_series_mismatch').sum()
        n_exact_series_match = ((out_df['match_type'] == 'exact') & (out_df['series_match'] == 'match')).sum()
        n_exact_series_unknown = ((out_df['match_type'] == 'exact') & (out_df['series_match'] == 'unknown')).sum()
        is_cosine_family = out_df['match_type'].isin(['cosine', 'cosine_reranked_by_series_position'])
        n_cosine_above = (is_cosine_family & out_df['above_experimental_threshold']).sum()
        n_cosine_below = (is_cosine_family & ~out_df['above_experimental_threshold']).sum()
        n_reranked = (out_df['match_type'] == 'cosine_reranked_by_series_position').sum()
        n_flagged = out_df['low_coherence_flag'].sum()
        print(f"  Exact matches accepted: {n_exact} (confirmed same-series: {n_exact_series_match}, "
              f"series unverifiable: {n_exact_series_unknown})")
        print(f"  Exact matches REJECTED (title string matched, but confirmed different series): {n_exact_rejected}")
        print(f"  Cosine matches >= {EXPERIMENTAL_THRESHOLD} (including {n_reranked} reranked by series position): {n_cosine_above}")
        print(f"  Cosine matches < {EXPERIMENTAL_THRESHOLD} (no match recorded): {n_cosine_below}")
        print(f"  Flagged low-coherence (accepted match, but central-vs-matched similarity < {EXPERIMENTAL_THRESHOLD}): {n_flagged}")
        print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()