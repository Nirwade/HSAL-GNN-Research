"""
Colab cells that built the embeddings and metadata files in this folder (provenance; they point at Google Drive paths).

Extracted verbatim from the Colab notebook Goodreads-2.ipynb (cells 97 and 99). Cell 102 (a reusable
generate_embeddings() whose __main__ wrote *_author_v2 files) is left out because those files are not part of this repo.
Cells are separated by '# ===== CELL n =====' markers. Run them in Colab, not as one script.

  97  Version A: title + description[:500] + genre  -> Goodreads_young_adult.{text,item2index,feat.npy} and the A metadata
  99  Version B: A plus the author name            -> Goodreads_young_adult_author.{text,item2index,feat.npy,meta.csv}
      (cell 97 needs GENRE = 'young_adult' defined first, as in upstream/build_hsal_dataset_colab_cells.py cell 40)

Inputs: inputs/Goodreads_young_adult_HSAL_100k.csv plus the raw UCSD files goodreads_books_young_adult.json.gz and
goodreads_book_authors.json.gz (see the README).

Verification of Version B (cell 99):
  .text and .meta.csv   rebuilt from the raw UCSD files: byte-identical to the committed files.
  .feat.npy             re-encoded from the committed .text with all-MiniLM-L6-v2 (sentence-transformers 6.2.0, torch 2.15.0, CPU,
                        batch size 256): max abs diff 3.0e-07, row cosine min 0.9999999, allclose at atol 1e-5. Equal up to float
                        rounding, not bit for bit. Version A (cell 97) was not re-run.
"""


# ===== CELL 97 =====
import gzip, json, math
import numpy as np
import pandas as pd

import os
print(os.path.exists(BOOKS_FILE))   # should print True
print(os.path.exists(FINAL_CSV))    # should print True

SAVE_DIR   = '/content/drive/MyDrive/HSAL_Goodreads/'
BOOKS_FILE = '/content/drive/MyDrive/Goodreads/goodreads_books_young_adult.json.gz'
FINAL_CSV  = SAVE_DIR + 'Goodreads_young_adult_HSAL_100k.csv'



# >>> EDIT THIS to the exact path of your young_adult HSAL interactions file on Drive <
FINAL_CSV  = SAVE_DIR + 'Goodreads_young_adult_HSAL_100k.csv'

TEXT_FILE  = SAVE_DIR + f'Goodreads_{GENRE}.text'
META_FILE  = SAVE_DIR + f'Goodreads_{GENRE}_item_meta_with_title.csv'
FEAT_FILE  = SAVE_DIR + f'Goodreads_{GENRE}.feat.npy'
IDX_FILE   = SAVE_DIR + f'Goodreads_{GENRE}.item2index'

# ── STEP 1: active item_ids from your HSAL interactions file ────────
print(f"[1] Loading active items from {FINAL_CSV}...")
df_final = pd.read_csv(FINAL_CSV)
active_items = set(df_final['item_id'].unique())
print(f"  Active items: {len(active_items):,}")

# ── STEP 2: extract title/description/genre for active items only ───
print(f"\n[2] Extracting metadata from {BOOKS_FILE}...")

SKIP_SHELVES = {
    'to-read', 'currently-reading', 'owned', 'default',
    'favorites', 'books-i-own', 'ebook', 'kindle',
    'library', 'audiobook', 'owned-books', 'book-club',
    'read', 'dnf', 'wish-list', 'novels', 'fiction',
    'non-fiction', 'books', 'my-books', 'nook', 'series',
    'adult', 'english', 'paperback', 'hardcover'
}

local_book_meta = {}
with gzip.open(BOOKS_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        bid = int(obj['book_id'])
        if bid not in active_items:
            continue

        title       = obj.get('title', '').strip()
        description = obj.get('description', '').strip()[:500]

        shelves = obj.get('popular_shelves', [])
        genre = ''
        for shelf in shelves:
            name = shelf.get('name', '').lower().strip()
            if name not in SKIP_SHELVES and len(name) > 2:
                genre = name
                break

        local_book_meta[bid] = {
            'title': title,
            'description': description,
            'genre': genre if genre else 'unknown',
        }

print(f"  Extracted metadata for {len(local_book_meta):,} active items")
missing = active_items - set(local_book_meta.keys())
print(f"  Items not found in books file: {len(missing):,}")

# ── STEP 3: sequel map from item_series_id (same logic as before) ───
print(f"\n[3] Parsing series positions...")

def parse_series_and_pos(series_id):
    if str(series_id).startswith('SB'):
        return None, 1
    parts = str(series_id).rsplit('_P', 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except:
            return parts[0], 1
    return str(series_id), 1

df_final[['series_name', 'pos']] = df_final['item_series_id'].apply(
    lambda x: pd.Series(parse_series_and_pos(x))
)

sequel_map = {}
series_groups = df_final[df_final['series_name'].notna()].groupby('series_name')
for series_name, group in series_groups:
    group_unique = group.drop_duplicates('item_id').sort_values('pos')
    item_ids_ordered = group_unique['item_id'].tolist()
    for idx, item_id in enumerate(item_ids_ordered):
        sequel_map[item_id] = item_ids_ordered[idx + 1:]

# ── STEP 4: write TEXT_FILE (this is the title/description file) ────
print(f"\n[4] Writing {TEXT_FILE}...")

written, skipped = 0, 0
with open(TEXT_FILE, 'w', encoding='utf-8') as f:
    for item_id in sorted(active_items):
        meta = local_book_meta.get(item_id, {})
        title = meta.get('title', '').strip()
        description = meta.get('description', '').strip()
        genre = meta.get('genre', '').strip()
        if not title:
            skipped += 1
            continue
        text = ' '.join(filter(None, [title, description, genre]))
        text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        f.write(f"{item_id}\t{text}\n")
        written += 1

print(f"  Written: {written:,}   Skipped (no title): {skipped:,}")

# ── STEP 5: write a title-visible meta CSV for your own inspection ──
print(f"\n[5] Writing {META_FILE}...")
rows = []
for item_id in sorted(active_items):
    meta = local_book_meta.get(item_id, {})
    rows.append({
        'item_id': item_id,
        'title': meta.get('title', ''),
        'i_genre_c': meta.get('genre', 'unknown'),
        'r_sequel': sequel_map.get(item_id, []),
    })
pd.DataFrame(rows).to_csv(META_FILE, index=False)
print(f"  Total items: {len(rows):,}")

# ── STEP 6: generate SentenceBERT embeddings (same model as before) ─
print(f"\n[6] Generating embeddings...")
try:
    from sentence_transformers import SentenceTransformer
except:
    import subprocess
    subprocess.call(['pip', 'install', 'sentence-transformers', '-q'])
    from sentence_transformers import SentenceTransformer

item_ids_ordered, texts = [], []
with open(TEXT_FILE, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            item_ids_ordered.append(int(parts[0]))
            texts.append(parts[1])

print(f"  Items to embed: {len(texts):,}")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

np.save(FEAT_FILE, embeddings)
with open(IDX_FILE, 'w') as f:
    for idx, item_id in enumerate(item_ids_ordered):
        f.write(f"{item_id}\t{idx}\n")

print(f"\nDone. Saved:")
print(f"  {TEXT_FILE}")
print(f"  {META_FILE}  (this one has titles — open it and sanity-check a few rows)")
print(f"  {FEAT_FILE}  shape={embeddings.shape}")
print(f"  {IDX_FILE}")


# ===== CELL 99 =====
import gzip, json
import numpy as np
import pandas as pd
import os

SAVE_DIR     = '/content/drive/MyDrive/HSAL_Goodreads/'
BOOKS_FILE   = '/content/drive/MyDrive/Goodreads/goodreads_books_young_adult.json.gz'
FINAL_CSV    = SAVE_DIR + 'Goodreads_young_adult_HSAL_100k.csv'
AUTHORS_FILE = SAVE_DIR + 'goodreads_book_authors.json.gz'

# ── Version B output files — exact names as specified ────────────────
TEXT_FILE_B  = SAVE_DIR + 'Goodreads_young_adult_author.text'
META_FILE_B  = SAVE_DIR + 'Goodreads_young_adult_author.meta.csv'
FEAT_FILE_B  = SAVE_DIR + 'Goodreads_young_adult_author.feat.npy'
IDX_FILE_B   = SAVE_DIR + 'Goodreads_young_adult_author.item2index'

print(os.path.exists(BOOKS_FILE), os.path.exists(FINAL_CSV), os.path.exists(AUTHORS_FILE))

# ── STEP 0: author_id -> name lookup ──────────────────────────────────
print(f"[0] Loading author names from {AUTHORS_FILE}...")
author_name_map = {}
with gzip.open(AUTHORS_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        author_name_map[str(obj['author_id'])] = obj.get('name', '').strip()
print(f"  Loaded {len(author_name_map):,} author names")

# ── STEP 1: active item_ids from HSAL interactions file ───────────────
df_final = pd.read_csv(FINAL_CSV)
active_items = set(df_final['item_id'].unique())
print(f"[1] Active items: {len(active_items):,}")

# ── STEP 2: sequel map (same derivation as Version A, kept for mapping back to HSAL) ─
def parse_series_and_pos(series_id):
    if str(series_id).startswith('SB'):
        return None, 1
    parts = str(series_id).rsplit('_P', 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except:
            return parts[0], 1
    return str(series_id), 1

df_final[['series_name', 'pos']] = df_final['item_series_id'].apply(
    lambda x: pd.Series(parse_series_and_pos(x))
)

sequel_map = {}
series_groups = df_final[df_final['series_name'].notna()].groupby('series_name')
for series_name, group in series_groups:
    group_unique = group.drop_duplicates('item_id').sort_values('pos')
    item_ids_ordered = group_unique['item_id'].tolist()
    for idx, item_id in enumerate(item_ids_ordered):
        sequel_map[item_id] = item_ids_ordered[idx + 1:]

# ── STEP 3: extract title, description, genre, author, rating ────────
print(f"\n[3] Extracting metadata from {BOOKS_FILE}...")

SKIP_SHELVES = {
    'to-read', 'currently-reading', 'owned', 'default',
    'favorites', 'books-i-own', 'ebook', 'kindle',
    'library', 'audiobook', 'owned-books', 'book-club',
    'read', 'dnf', 'wish-list', 'novels', 'fiction',
    'non-fiction', 'books', 'my-books', 'nook', 'series',
    'adult', 'english', 'paperback', 'hardcover'
}

local_book_meta = {}
with gzip.open(BOOKS_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        obj = json.loads(line)
        bid = int(obj['book_id'])
        if bid not in active_items:
            continue

        title       = obj.get('title', '').strip()
        description = obj.get('description', '').strip()[:500]

        shelves = obj.get('popular_shelves', [])
        genre = ''
        for shelf in shelves:
            name = shelf.get('name', '').lower().strip()
            if name not in SKIP_SHELVES and len(name) > 2:
                genre = name
                break

        author_ids   = [a.get('author_id', '') for a in obj.get('authors', [])]
        author_names = [author_name_map.get(aid, aid) for aid in author_ids]  # falls back to ID if unresolved

        local_book_meta[bid] = {
            'title': title,
            'description': description,
            'genre': genre if genre else 'unknown',
            'authors': '; '.join(author_names) if author_names else '',
            'average_rating': obj.get('average_rating', ''),
            'ratings_count': obj.get('ratings_count', ''),
        }

print(f"  Extracted for {len(local_book_meta):,} items")
missing = active_items - set(local_book_meta.keys())
print(f"  Items not found in books file: {len(missing):,}")

# ── STEP 4: TEXT_FILE_B — embedding text = title + description + genre + author ─
print(f"\n[4] Writing {TEXT_FILE_B}...")

written, skipped = 0, 0
with open(TEXT_FILE_B, 'w', encoding='utf-8') as f:
    for item_id in sorted(active_items):
        meta = local_book_meta.get(item_id, {})
        title = meta.get('title', '').strip()
        description = meta.get('description', '').strip()
        genre = meta.get('genre', '').strip()
        authors = meta.get('authors', '').strip()

        if not title:
            skipped += 1
            continue

        # rating deliberately excluded here — numeric, not semantic text
        text = ' '.join(filter(None, [title, description, genre, authors]))
        text = text.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
        f.write(f"{item_id}\t{text}\n")
        written += 1

print(f"  Written: {written:,}   Skipped (no title): {skipped:,}")

# ── STEP 5: META_FILE_B — full metadata incl. rating + HSAL mapping fields ─
print(f"\n[5] Writing {META_FILE_B}...")
rows = []
for item_id in sorted(active_items):
    meta = local_book_meta.get(item_id, {})
    rows.append({
        'item_id': item_id,
        'title': meta.get('title', ''),
        'description': meta.get('description', ''),
        'authors': meta.get('authors', ''),
        'average_rating': meta.get('average_rating', ''),
        'ratings_count': meta.get('ratings_count', ''),
        'i_genre_c': meta.get('genre', 'unknown'),
        'r_sequel': sequel_map.get(item_id, []),
    })
pd.DataFrame(rows).to_csv(META_FILE_B, index=False)
print(f"  Total items: {len(rows):,}")

# ── STEP 6: embeddings — same model as Version A, separate output files ─
print(f"\n[6] Generating Version B embeddings...")
try:
    from sentence_transformers import SentenceTransformer
except:
    import subprocess
    subprocess.call(['pip', 'install', 'sentence-transformers', '-q'])
    from sentence_transformers import SentenceTransformer

item_ids_ordered, texts = [], []
with open(TEXT_FILE_B, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            item_ids_ordered.append(int(parts[0]))
            texts.append(parts[1])

print(f"  Items to embed: {len(texts):,}")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)

np.save(FEAT_FILE_B, embeddings)
with open(IDX_FILE_B, 'w') as f:
    for idx, item_id in enumerate(item_ids_ordered):
        f.write(f"{item_id}\t{idx}\n")

print(f"\nDone — Version B (Version A untouched):")
print(f"  {TEXT_FILE_B}")
print(f"  {META_FILE_B}")
print(f"  {FEAT_FILE_B}  shape={embeddings.shape}")
print(f"  {IDX_FILE_B}")
