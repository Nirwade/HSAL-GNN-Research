"""
Validate the newly regenerated Version B (_v2) embeddings against the
original Version B before trusting or replacing anything.

Checks:
  1. Same set of item_ids present in both.
  2. Coherence (cosine similarity between two books' own embeddings) on the
     same known-correct relationship pairs already validated for Version A
     tonight — if old-B and new-B give similar coherence scores on these,
     that's real evidence the two pipelines produce comparable embeddings.
  3. A quick diff of the meta CSVs (author/genre/rating) for the same items,
     since two separate runs of shelf-based genre extraction could pick a
     different shelf if Goodreads' shelf ordering/counts shifted between runs.
"""

import numpy as np
import pandas as pd

SAVE_DIR = './'

OLD_FEAT = SAVE_DIR + 'Goodreads_young_adult_author.feat.npy'
OLD_IDX = SAVE_DIR + 'Goodreads_young_adult_author.item2index'
OLD_META = SAVE_DIR + 'Goodreads_young_adult_author.meta.csv'

NEW_FEAT = SAVE_DIR + 'Goodreads_young_adult_author_v2.feat.npy'
NEW_IDX = SAVE_DIR + 'Goodreads_young_adult_author_v2.item2index'
NEW_META = SAVE_DIR + 'Goodreads_young_adult_author_v2.meta.csv'

# Same known-correct relationship pairs validated on Version A tonight —
# (central_item_id, related_item_id, description)
KNOWN_PAIRS = [
    (1842609, 219563, "The House of Djinn -> Shabanu: Daughter of the Wind (prequel, confirmed)"),
    (30694168, 24763621, "Shadowsong -> Wintersong (prequel, confirmed)"),
    (12663646, 11614718, "Hana -> Delirium (prequel, confirmed)"),
]


def load_index(path):
    idx = {}
    with open(path, 'r') as f:
        for line in f:
            item_id, i = line.strip().split('\t')
            idx[int(item_id)] = int(i)
    return idx


def coherence(embeddings, item2index, id_a, id_b):
    ia, ib = item2index.get(id_a), item2index.get(id_b)
    if ia is None or ib is None:
        return None
    va, vb = embeddings[ia], embeddings[ib]
    return float((va @ vb) / ((np.linalg.norm(va) + 1e-10) * (np.linalg.norm(vb) + 1e-10)))


def main():
    old_feat = np.load(OLD_FEAT)
    old_idx = load_index(OLD_IDX)
    new_feat = np.load(NEW_FEAT)
    new_idx = load_index(NEW_IDX)

    print(f"Old: {old_feat.shape}, {len(old_idx)} ids")
    print(f"New: {new_feat.shape}, {len(new_idx)} ids")

    old_ids, new_ids = set(old_idx.keys()), set(new_idx.keys())
    only_old = old_ids - new_ids
    only_new = new_ids - old_ids
    print(f"\nItem_id overlap: {len(old_ids & new_ids)} / {len(old_ids | new_ids)}")
    if only_old:
        print(f"  In OLD only ({len(only_old)}): {list(only_old)[:5]}")
    if only_new:
        print(f"  In NEW only ({len(only_new)}): {list(only_new)[:5]}")

    print("\n=== Coherence on known-correct pairs (old-B vs new-B) ===")
    for id_a, id_b, desc in KNOWN_PAIRS:
        c_old = coherence(old_feat, old_idx, id_a, id_b)
        c_new = coherence(new_feat, new_idx, id_a, id_b)
        diff = abs(c_old - c_new) if (c_old is not None and c_new is not None) else None
        print(f"  {desc}")
        print(f"    old={c_old}   new={c_new}   |diff|={diff}")

    print("\n=== Meta CSV spot-check (same items, old vs new) ===")
    old_meta = pd.read_csv(OLD_META).set_index('item_id')
    new_meta = pd.read_csv(NEW_META).set_index('item_id')
    for id_a, id_b, desc in KNOWN_PAIRS:
        for iid in (id_a, id_b):
            if iid in old_meta.index and iid in new_meta.index:
                o, n = old_meta.loc[iid], new_meta.loc[iid]
                print(f"  item {iid}:")
                print(f"    old authors='{o.get('authors','')}'  genre='{o.get('i_genre_c','')}'")
                print(f"    new authors='{n.get('authors','')}'  genre='{n.get('i_genre_c','')}'")


if __name__ == "__main__":
    main()