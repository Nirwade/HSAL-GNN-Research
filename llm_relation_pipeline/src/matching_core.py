"""
Shared matching logic, validated tonight against the Prequel pipeline:
  - title normalization + exact match
  - series-name parsing and identity check (rejects title collisions across
    different series, e.g. "Marked" existing in two unrelated series)
  - series-position parsing (bare integers only — '#1', '#2', not '#1.5'/'#0')
  - coherence check (corpus-embedding similarity between two books' own
    vectors) — used as a secondary signal after exact match, since it's the
    stronger, independent check there; NOT applied to confirmed same-series
    exact matches, where series identity already outranks a thin-description
    embedding score
  - cosine similarity fallback + series-aware rerank, parameterized by
    `direction`: 'before' for prequel (candidate position < central),
    'after' for sequel (candidate position > central)

Any relation type whose "correct" match is a directional neighbor in the same
series (prequel, sequel, midquel) can reuse this as-is via the direction
parameter. Relation types with no positional meaning (spin-off, retelling,
alternative_version) should skip the position-based rerank and rely on exact
match + cosine + coherence only — see each extractor's own driver script.
"""

import re

import numpy as np

EXPERIMENTAL_THRESHOLD = 0.5  # not a final cutoff — validated against real
                               # cases for Prequel/Version A; re-validate
                               # before trusting it for a new relation type
                               # or embedding version.

RERANK_POSITION_WINDOW = 3  # only rerank within this many positions of the
                             # central book — prevents "any earlier/later
                             # book in a 90-book episodic series" false fixes

SERIES_SUFFIX_RE = re.compile(r'\s*\([^)]*\)\s*$')
PUNCT_RE = re.compile(r'[^a-z0-9\s]')
SERIES_POS_RE = re.compile(r'\(([^,()]+),\s*#(\d+)\)\s*$')
SERIES_NAME_RE = re.compile(r'\(([^()]+)\)\s*$')


def normalize_title(title):
    t = title.strip()
    t = SERIES_SUFFIX_RE.sub('', t)
    t = t.lower()
    t = PUNCT_RE.sub('', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def parse_series_position(title):
    """Bare-integer position only ('#1', '#37') — excludes '#1.5', '#4.4',
    '#0' (novellas/interstitials/prequel-markers, not reliable ordering)."""
    m = SERIES_POS_RE.search(title.strip())
    if not m:
        return None, None
    series_raw, pos_str = m.group(1), m.group(2)
    pos = int(pos_str)
    if pos == 0:
        return None, None
    series_norm = re.sub(r'\s+', ' ', PUNCT_RE.sub('', series_raw.lower())).strip()
    return series_norm, pos


def parse_series_name(title):
    """Series name only, regardless of whether the position is a bare
    integer, a decimal, or absent. Returns None if unparseable — that's
    'unknown', not 'mismatch': absence of a label proves nothing either way."""
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


def make_coherence_fn(embeddings, item2index):
    """Returns a coherence(id_a, id_b) function bound to a specific corpus's
    embeddings/index, so callers don't have to thread both through everywhere."""
    def coherence(id_a, id_b):
        idx_a, idx_b = item2index.get(id_a), item2index.get(id_b)
        if idx_a is None or idx_b is None:
            return None
        va, vb = embeddings[idx_a], embeddings[idx_b]
        return float(
            (va @ vb) / ((np.linalg.norm(va) + 1e-10) * (np.linalg.norm(vb) + 1e-10))
        )
    return coherence


def resolve_exact_match(central_id, central_title, candidates, id_to_title):
    """Among title-string candidates, prefer one confirmed to share the
    central book's series name; fall back to an unverifiable one (no
    parseable series on one side); reject outright only if every candidate
    is a CONFIRMED different series (a coincidental title collision).

    Returns (chosen_id_or_None, series_status, all_candidate_info)."""
    central_series_name = parse_series_name(central_title)
    cand_info = []
    for cid in candidates:
        cand_series_name = parse_series_name(id_to_title.get(cid, ''))
        if central_series_name is None or cand_series_name is None:
            status = 'unknown'
        elif central_series_name == cand_series_name:
            status = 'match'
        else:
            status = 'mismatch'
        cand_info.append((cid, status))

    chosen = next((c for c in cand_info if c[1] == 'match'), None)
    if chosen is None:
        chosen = next((c for c in cand_info if c[1] == 'unknown'), None)

    if chosen is not None:
        return chosen[0], chosen[1], cand_info
    return None, 'mismatch', cand_info


def rerank_by_series_position(central_title, top5, id_to_title, direction):
    """direction='before' for prequel (candidate position < central),
    direction='after' for sequel (candidate position > central).
    Only considers candidates within RERANK_POSITION_WINDOW of the central
    book's position, and only within the given top5 — does not scan the
    full corpus. Returns (item_id, score, used_bool) — falls back to
    top5[0] unchanged if nothing qualifies."""
    central_series, central_pos = parse_series_position(central_title)
    if not central_series or not central_pos:
        return top5[0][0], top5[0][1], False

    qualifying = []
    for cand_id, cand_score in top5:
        cand_series, cand_pos = parse_series_position(id_to_title.get(cand_id, ''))
        if cand_series != central_series or cand_pos is None:
            continue
        if direction == 'before' and cand_pos < central_pos and (central_pos - cand_pos) <= RERANK_POSITION_WINDOW:
            qualifying.append((cand_id, cand_score, cand_pos))
        elif direction == 'after' and cand_pos > central_pos and (cand_pos - central_pos) <= RERANK_POSITION_WINDOW:
            qualifying.append((cand_id, cand_score, cand_pos))

    if not qualifying:
        return top5[0][0], top5[0][1], False

    if direction == 'before':
        qualifying.sort(key=lambda x: -x[2])
    else:
        qualifying.sort(key=lambda x: x[2])
    chosen_id, chosen_score, _ = qualifying[0]
    return chosen_id, chosen_score, True