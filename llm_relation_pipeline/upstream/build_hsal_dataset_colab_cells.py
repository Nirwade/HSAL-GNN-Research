"""
PROVENANCE ONLY. Not re-run and not verified: it needs the multi-GB raw Goodreads files and a Colab
environment, and the 100k file comes from random sampling. The resulting file is committed as
inputs/Goodreads_young_adult_HSAL_100k.csv, which is what the rest of the pipeline uses.

Extracted verbatim from the Colab notebook Goodreads-2.ipynb (cells 36, 37, 40, 41, 42, 43, 44).
Cells are separated by '# ===== CELL n =====' markers. Run them in order in Colab with GENRE = 'young_adult'
(cell 40 sets it). Paths point at Google Drive (/content/drive/MyDrive/HSAL_Goodreads/).

What the cells do:
  36-37  mount Drive, download the per-genre books files from the UCSD Goodreads page
  40-41  pick the genre, set the target row count and tier budgets, download the interactions file
  42     build franchise_map.json from the full books file (collapses editions of one franchise into one series)
  43     keep English US canonical editions, reservoir-sample interactions with sqrt-weighted per-series budgets,
         resolve publication dates, assign series positions, write Goodreads_YoungAdult_HSAL_Final.csv
         with item_series_id = Series_<id>_P<n> or SB<id> (standalone)
  44     subsample each genre's Final file to 100k rows -> Goodreads_<genre>_HSAL_100k.csv
"""


# ===== CELL 36 =====
from google.colab import drive
drive.mount('/content/drive')

import os
SAVE_DIR = '/content/drive/MyDrive/HSAL_Goodreads/'
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"✓ Drive mounted — saving to {SAVE_DIR}")


# ===== CELL 37 =====


BOOKS_URLS = {
    'fantasy':   'goodreads_books_fantasy_paranormal.json.gz',
    'young_adult': 'goodreads_books_young_adult.json.gz',
    'mystery':   'goodreads_books_mystery_thriller_crime.json.gz',
    'romance':   'goodreads_books_romance.json.gz',
    'history':   'goodreads_books_history_biography.json.gz',
    'children':  'goodreads_books_children.json.gz',
    'comics':    'goodreads_books_comics_graphic.json.gz',
    'poetry':    'goodreads_books_poetry.json.gz',
}

BASE_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/"

for genre, filename in BOOKS_URLS.items():
    dest = f"{SAVE_DIR}{filename}"
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"✓ Already exists — {filename}  ({size_mb:.1f} MB)")
    else:
        print(f"Downloading {filename}...")
        os.system(f'wget --show-progress -c "{BASE_URL}{filename}" -O "{dest}"')
        size_mb = os.path.getsize(dest) / 1024 / 1024
        print(f"✓ Saved — {size_mb:.1f} MB")


# ===== CELL 40 =====
# Change GENRE to switch between runs
#GENRE = 'mystery'
#GENRE = 'fantasy_paranormal'
GENRE = 'young_adult'
#GENRE = 'romance'

GENRE_CONFIG = {
    'mystery': {
        'books_file':        'goodreads_books_mystery_thriller_crime.json.gz',
        'interactions_file': 'goodreads_interactions_mystery_thriller_crime.json.gz',
        'target_rows':       500_000,
        'label':             'Mystery, Thriller & Crime',
    },
    'fantasy_paranormal': {
        'books_file':        'goodreads_books_fantasy_paranormal.json.gz',
        'interactions_file': 'goodreads_interactions_fantasy_paranormal.json.gz',
        'target_rows':       700_000,
        'label':             'Fantasy & Paranormal',
    },
    'young_adult': {
        'books_file':        'goodreads_books_young_adult.json.gz',
        'interactions_file': 'goodreads_interactions_young_adult.json.gz',
        'target_rows':       300_000,
        'label':             'Young Adult',
    },
    'romance': {
        'books_file':        'goodreads_books_romance.json.gz',
        'interactions_file': 'goodreads_interactions_romance.json.gz',
        'target_rows':       500_000,
        'label':             'Romance',
    },
}

SAVE_DIR = '/content/drive/MyDrive/HSAL_Goodreads/'

cfg      = GENRE_CONFIG[GENRE]

BOOKS_FILE        = SAVE_DIR + cfg['books_file']
INTERACTIONS_FILE = SAVE_DIR + cfg['interactions_file']
TARGET_ROWS       = cfg['target_rows']
GENRE_LABEL       = cfg['label']

# output one final HSAL-ready file per genre
# Explicit filename map — matches actual files on Drive
FINAL_CSV_MAP = {
    'mystery':            'Goodreads_Mystery_HSAL_Final.csv',
    'fantasy_paranormal': 'Goodreads_Fantasy_HSAL_Final.csv',
    'young_adult':        'Goodreads_YoungAdult_HSAL_Final.csv',
    'romance':            'Goodreads_Romance_HSAL_Final.csv',
}
FINAL_CSV = SAVE_DIR + FINAL_CSV_MAP[GENRE]

# tier row allocation: 60% high-signal, 30% mid-signal, 10% standalone
TIER1_ROWS  = int(TARGET_ROWS * 0.60)
TIER2_ROWS  = int(TARGET_ROWS * 0.30)
TIER3_ROWS  = int(TARGET_ROWS * 0.10)
TIER1_MIN   = 1_000
TIER2_MIN   = 100
TIER3_TOP_N = 500

import os
print(f"Genre   : {GENRE_LABEL}")
print(f"Target  : {TARGET_ROWS:,} rows")
print(f"Output  : {FINAL_CSV}")


# ===== CELL 41 =====
BASE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/'

dest = SAVE_DIR + cfg['interactions_file']
if os.path.exists(dest):
    print(f"{cfg['interactions_file']} already on Drive ({os.path.getsize(dest)/1024/1024:.0f} MB)")
else:
    print(f"Downloading {cfg['interactions_file']}...")
    os.system(f'wget --show-progress -c "{BASE_URL}{cfg["interactions_file"]}" -O "{dest}"')
    print(f"Done — {os.path.getsize(dest)/1024/1024:.0f} MB")


# ===== CELL 42 =====
# ─────────────────────────────────────────────────────────────────
# FRANCHISE MAP BUILDER
#
#
# Output: franchise_map.json
#   {book_id: {
#       'canonical_series_id': '167817',  ← HP#1's series_id
#       'title_position': 1.0,            ← from title (#1, #2...)
#       'franchise_name': 'harry potter'  ← normalized
#   }}
# ─────────────────────────────────────────────────────────────────
import gzip, json, re, os
from collections import defaultdict

SAVE_DIR    = '/content/drive/MyDrive/HSAL_Goodreads/'
FULL_BOOKS  = SAVE_DIR + 'goodreads_books.json.gz'
OUTPUT_FILE = SAVE_DIR + 'franchise_map.json'

if os.path.exists(OUTPUT_FILE):
    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"franchise_map.json already exists ({size_kb:.0f} KB)")
    print("Delete it and re-run if you want to rebuild.")
else:
    print("Building franchise map from full books file...")
    print("This runs ONCE and takes ~5-10 min")

    # ── Normalize franchise name ──────────────────────────────────
    def normalize_franchise(raw):
        name = raw.strip()
        name = re.sub(r',?\s*#[\d.]+.*$', '', name).strip()
        for suffix in [' series', ' trilogy', ' saga',
                       ' chronicles', ' collection']:
            if name.lower().endswith(suffix):
                name = name[:-len(suffix)].strip()
        name = re.sub(
            r'^(the|a|an)\s+', '', name, flags=re.IGNORECASE
        ).strip()
        name = name.lower()
        name = re.sub(r'[^a-z0-9\s]', '', name).strip()
        name = re.sub(r'\s+', ' ', name)
        return name

    def extract_franchise_and_pos(title):
    # Only match simple positions: #1, #2, #0.5 — NOT #3.5.2
        match = re.search(r'\((.+?),?\s*#(\d+(?:\.\d)?)\)', title)
        if match:
            franchise = normalize_franchise(match.group(1))
            try:
                pos = float(match.group(2))
            except ValueError:
                return None, None
            return franchise, pos
        return None, None

    # ── Pass 1: collect all books per franchise ───────────────────
    # For each franchise, find which series_id belongs to position #1
    # That becomes the canonical series_id for the whole franchise
    print(" Pass 1: collecting franchise data...")

    # {franchise_name: [(pos, series_id, book_id, ratings)]}
    franchise_data = defaultdict(list)
    # {book_id: (franchise_name, pos, series_id)}
    book_franchise = {}

    n = 0
    with gzip.open(FULL_BOOKS, 'rt', encoding='utf-8') as f:
        for line in f:
            obj     = json.loads(line)
            bid     = int(obj['book_id'])
            title   = obj.get('title', '')
            series  = obj.get('series', [])
            lang    = obj.get('language_code', '')
            country = obj.get('country_code', '')
            ratings = int(obj.get('ratings_count', 0) or 0)

            if not series:
                continue
            if lang not in ['eng', 'en-US']:
                continue
            if country != 'US':
                continue

            franchise, pos = extract_franchise_and_pos(title)
            if not franchise or pos is None:
                continue

            sid = str(series[0])
            franchise_data[franchise].append((pos, sid, bid, ratings))
            book_franchise[bid] = (franchise, pos, sid)

            n += 1
            if n % 500_000 == 0:
                print(f"  {n/1e6:.1f}M books scanned")

    print(f"  {n:,} books with franchise+position info")
    print(f"  {len(franchise_data):,} unique franchises found")

    # ── Pass 2: find canonical series_id per franchise ────────────
    # Rule: canonical = series_id of the book at position #1
    # If multiple books at #1, use highest ratings
    print(" Pass 2: finding canonical series_id per franchise...")

    franchise_canonical = {}
    # {franchise_name: canonical_series_id}

    for franchise, books in franchise_data.items():
        # Find all books at position 1
        pos1_books = [(sid, bid, rc) for pos, sid, bid, rc in books
                      if pos == 1.0]
        if not pos1_books:
            # No book at position 1 — use lowest position
            min_pos = min(pos for pos, _, _, _ in books)
            pos1_books = [(sid, bid, rc) for pos, sid, bid, rc in books
                          if pos == min_pos]

        # Pick highest rated book at position 1
        best = max(pos1_books, key=lambda x: x[2])
        canonical_sid = best[0]
        franchise_canonical[franchise] = canonical_sid

    print(f"  {len(franchise_canonical):,} canonical series_ids assigned")

    # ── Build final franchise_map ─────────────────────────────────
    # {book_id: {canonical_series_id, title_position, franchise_name}}
    franchise_map = {}

    for bid, (franchise, pos, original_sid) in book_franchise.items():
        canonical_sid = franchise_canonical.get(franchise)
        if canonical_sid:
            franchise_map[str(bid)] = {
                'canonical_series_id': canonical_sid,
                'title_position':      pos,
                'franchise_name':      franchise,
            }

    # Save to Drive
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(franchise_map, f)

    size_kb = os.path.getsize(OUTPUT_FILE) / 1024
    print(f"\nSaved: {OUTPUT_FILE}  ({size_kb:.0f} KB)")
    print(f"Total books mapped: {len(franchise_map):,}")

    # Quick verification — Harry Potter
    print("\nVerification — Harry Potter:")
    hp_check = {
        bid: data for bid, data in franchise_map.items()
        if data['franchise_name'] == 'harry potter'
    }
    print(f"  HP books in map: {len(hp_check)}")
    for bid, data in sorted(
        hp_check.items(),
        key=lambda x: x[1]['title_position']
    )[:10]:
        print(f"  #{data['title_position']}: "
              f"book_id={bid} "
              f"canonical_sid={data['canonical_series_id']}")


# ===== CELL 43 =====
import os, json, gzip, math, random
from collections import defaultdict
from datetime import datetime, date
import calendar
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

print(f"\n{GENRE_LABEL}")
print("-" * 40)


# Builds:
#   book_to_series : {book_id: Series_XXX or SB{bid}}
#   book_meta_raw  : {book_id: pub_year, pub_month, pub_day,
#                              ratings_count, work_id,
#                              language_code, country_code}

# Load franchise map built above
import json as _json
FRANCHISE_MAP_FILE = SAVE_DIR + 'franchise_map.json'
with open(FRANCHISE_MAP_FILE, 'r') as _f:
    franchise_map = _json.load(_f)
print(f"  Franchise map loaded: {len(franchise_map):,} books")

book_to_series = {}
book_meta_raw  = {}
book_title_pos = {}  # {book_id: float position from title, or None}

with gzip.open(BOOKS_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        obj    = json.loads(line)
        bid    = int(obj['book_id'])
        series = obj.get('series', [])

        # Use franchise map if available (franchise-level grouping)
        # Otherwise fall back to McAuley series_id (edition-level)
        fm = franchise_map.get(str(bid))
        if fm:
            book_to_series[bid] = f"Series_{fm['canonical_series_id']}"
            book_title_pos[bid] = fm['title_position']
        elif series:
            book_to_series[bid] = f"Series_{series[0]}"
            book_title_pos[bid] = None
        else:
            book_to_series[bid] = f"SB{bid}"
            book_title_pos[bid] = None

        year  = str(obj.get('publication_year',  ''))
        month = str(obj.get('publication_month', ''))
        day   = str(obj.get('publication_day',   ''))

        book_meta_raw[bid] = {
            'pub_year':      int(year)  if year.isdigit()  else None,
            'pub_month':     int(month) if month.isdigit() else None,
            'pub_day':       int(day)   if day.isdigit()   else None,
            'ratings_count': int(obj.get('ratings_count', 0) or 0),
            'work_id':       str(obj.get('work_id', '')),
            'language_code': obj.get('language_code', ''),
            'country_code':  obj.get('country_code', ''),
        }

total_books = len(book_to_series)
print(f"  {total_books:,} books read from file")


# Applies to BOTH ISQ (series) and ISA (standalone) books
# Rule: language=eng/en-US + country=US + highest ratings_count per work_id
# All other editions → completely removed (not demoted, not reassigned)
# This fixes:
#   1. Non-English editions removed from dataset
#   2. Duplicate US English editions deduplicated to one canonical per story
#   3. Sequel positions fixed automatically (one book per series slot)
print(" filtering canonical editions (ISQ + ISA)")

ALLOWED_LANGUAGES = {'eng', 'en-US'}
ALLOWED_COUNTRY   = 'US'

# Find canonical edition per work_id across ALL books
work_to_best = {}
# {work_id: (book_id, ratings_count)}

for bid, meta in book_meta_raw.items():
    lang    = meta['language_code']
    country = meta['country_code']
    wid     = meta['work_id']

    # must pass language + country filter
    if lang not in ALLOWED_LANGUAGES:
        continue
    if country != ALLOWED_COUNTRY:
        continue
    # must have a work_id
    if not wid:
        continue

    rc = meta['ratings_count']
    if wid not in work_to_best or rc > work_to_best[wid][1]:
        work_to_best[wid] = (bid, rc)

canonical_books = {bid for bid, _ in work_to_best.values()}

# Remove all non-canonical books completely
# Both series AND standalone go through this filter
removed_books = 0
for bid in list(book_to_series.keys()):
    if bid not in canonical_books:
        del book_to_series[bid]
        removed_books += 1

series_remaining   = sum(
    1 for v in book_to_series.values() if v.startswith('Series_')
)
standalone_remaining = sum(
    1 for v in book_to_series.values() if v.startswith('SB')
)

print(f"  Total books in file:          {total_books:,}")
print(f"  Canonical books kept:         {len(canonical_books):,}")
print(f"  Books removed:                {removed_books:,}")
print(f"  Series books remaining:       {series_remaining:,}")
print(f"  Standalone books remaining:   {standalone_remaining:,}")

# ── STEP 2: Pre-scan interaction counts ──────────────────────────
# Streams full interactions file once
# Counts interactions per series label
# Removed books → their interactions are silently skipped (Option A)
print(" pre-scan (3-5 min)")

series_counts = defaultdict(int)
n = 0
skipped_interactions = 0

with gzip.open(INTERACTIONS_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        interaction = json.loads(line)
        bid = int(interaction['book_id'])

        if bid not in book_to_series:
            skipped_interactions += 1
            continue

        sid = book_to_series[bid]
        series_counts[sid] += 1
        n += 1
        if n % 5_000_000 == 0:
            print(f"  {n/1e6:.0f}M counted")

series_counts = dict(series_counts)
print(f"  {n:,} interactions counted")
print(f"  {skipped_interactions:,} interactions skipped (removed books)")
print(f"  {len([k for k in series_counts if k.startswith('Series_')]):,} real series")

# ── STEP 3: Tier classification ───────────────────────────────────
print(" classifying tiers")

series_s = pd.Series(series_counts).sort_values(ascending=False)
tier1 = set(series_s[
    series_s.index.str.startswith('Series_') &
    (series_s >= TIER1_MIN)
].index)
tier2 = set(series_s[
    series_s.index.str.startswith('Series_') &
    (series_s >= TIER2_MIN) &
    (series_s < TIER1_MIN)
].index)
tier3 = set(series_s[
    series_s.index.str.startswith('SB')
].head(TIER3_TOP_N).index)

print(f"  Tier 1 ({TIER1_MIN}+): {len(tier1):,} series -> {TIER1_ROWS:,} rows")
print(f"  Tier 2 (100-999) : {len(tier2):,} series -> {TIER2_ROWS:,} rows")
print(f"  Tier 3 (top SB)  : {len(tier3):,} books  -> {TIER3_ROWS:,} rows")
print(f"  Excluded         : "
      f"{len(series_counts)-len(tier1)-len(tier2)-len(tier3):,}")


def compute_budgets(series_set, tier_rows):
    weights = {s: math.sqrt(series_counts[s]) for s in series_set}
    total_w = sum(weights.values())
    budgets = {
        s: max(1, round(w / total_w * tier_rows))
        for s, w in weights.items()
    }
    diff = tier_rows - sum(budgets.values())
    top  = sorted(budgets, key=lambda k: weights[k], reverse=True)
    for i in range(abs(diff)):
        budgets[top[i % len(top)]] += int(math.copysign(1, diff))
    return budgets

all_budgets = {
    **compute_budgets(tier1, TIER1_ROWS),
    **compute_budgets(tier2, TIER2_ROWS),
    **compute_budgets(tier3, TIER3_ROWS),
}
print(f"  Total budgeted: {sum(all_budgets.values()):,} rows")

# ── STEP 5: Reservoir sampling ───────────────────────────────────
# Streams full interactions file once
# Removed book interactions skipped
# User IDs local per genre (0,1,2...) — cross-genre re-index in Phase 3
print(" reservoir sampling (15-25 min)")

reservoirs   = {sid: [] for sid in all_budgets}
seen_per_sid = {}
user_map     = {}
user_counter = 0
n_scanned    = 0

with gzip.open(INTERACTIONS_FILE, 'rt', encoding='utf-8') as f:
    for line in f:
        interaction = json.loads(line)
        bid = int(interaction['book_id'])

        # skip removed books (Option A)
        if bid not in book_to_series:
            continue

        sid = book_to_series[bid]
        if sid not in all_budgets:
            continue

        budget = all_budgets[sid]
        seen_per_sid[sid] = seen_per_sid.get(sid, 0) + 1
        n = seen_per_sid[sid]

        raw_uid = interaction['user_id']
        if raw_uid not in user_map:
            user_map[raw_uid] = user_counter
            user_counter += 1

        date_str = interaction.get('date_updated', '')
        try:
            unix_time = int(
                datetime.strptime(
                    date_str, "%a %b %d %H:%M:%S %z %Y"
                ).timestamp()
            )
        except ValueError:
            unix_time = 0

        record = {
            'user_id':        user_map[raw_uid],
            'item_id':        bid,
            'time':           unix_time,
            'item_series_id': sid,
        }

        if len(reservoirs[sid]) < budget:
            reservoirs[sid].append(record)
        else:
            j = random.randint(0, n - 1)
            if j < budget:
                reservoirs[sid][j] = record

        n_scanned += 1
        if n_scanned % 5_000_000 == 0:
            collected = sum(len(v) for v in reservoirs.values())
            print(f"  scanned {n_scanned/1e6:.0f}M | collected {collected:,}")

all_records     = [r for res in reservoirs.values() for r in res]
random.shuffle(all_records)
df_interactions = pd.DataFrame(all_records)
active_books    = set(df_interactions['item_id'])

print(f"  {len(df_interactions):,} rows sampled")
print(f"  users: 0 to {df_interactions['user_id'].max():,} "
      f"({df_interactions['user_id'].nunique():,} unique)")

# ── STEP 6: Publication date resolution (6-case logic) ───────────
# Case 1: no pub_year → remove book
# Case 2: year only → (year, Jan, 1) find free slot in series
# Case 3: year + month, no day → (year, month, 1) find free slot
# Case 4: year + day, no month → (year, Jan, day) find free slot
#         respects calendar + leap years, wraps to next month
# Case 5: year + month + day → keep, find next slot if conflict
# Case 6: month + day but no year → remove book
# Slot conflicts checked within same series only
print(" resolving publication dates (6-case logic)")

def next_valid_date(year, month, day):
    """Increment date by 1, respecting calendar and leap years."""
    max_day = calendar.monthrange(year, month)[1]
    if day < max_day:
        return year, month, day + 1
    elif month < 12:
        return year, month + 1, 1
    else:
        return year + 1, 1, 1

def resolve_date(bid, used_slots):
    """
    Resolve pub date for book bid.
    used_slots: set of (year, month, day) already taken in this series.
    Returns (year, month, day) or None if book should be removed.
    """
    meta = book_meta_raw.get(bid, {})
    yr   = meta.get('pub_year')
    mo   = meta.get('pub_month')
    dy   = meta.get('pub_day')

    # Case 1: no year → remove
    if yr is None:
        return None

    # Case 6: month+day but no year → remove
    # (already caught by Case 1 since yr is None)

    # Case 5: all three → keep, resolve conflict if any
    if yr is not None and mo is not None and dy is not None:
        cy, cm, cd = yr, mo, dy
        while True:
            slot = (cy, cm, cd)
            if slot not in used_slots:
                return slot
            cy, cm, cd = next_valid_date(cy, cm, cd)

    # Case 2: year only
    if yr is not None and mo is None and dy is None:
        cy, cm, cd = yr, 1, 1
        while True:
            slot = (cy, cm, cd)
            if slot not in used_slots:
                return slot
            cy, cm, cd = next_valid_date(cy, cm, cd)

    # Case 3: year + month, no day
    if yr is not None and mo is not None and dy is None:
        cy, cm, cd = yr, mo, 1
        while True:
            slot = (cy, cm, cd)
            if slot not in used_slots:
                return slot
            cy, cm, cd = next_valid_date(cy, cm, cd)

    # Case 4: year + day, no month → use Jan
    if yr is not None and mo is None and dy is not None:
        # validate day against January
        max_jan = calendar.monthrange(yr, 1)[1]
        cd = min(dy, max_jan)
        cy, cm = yr, 1
        while True:
            slot = (cy, cm, cd)
            if slot not in used_slots:
                return slot
            cy, cm, cd = next_valid_date(cy, cm, cd)

    return None

# Resolve dates for all active books
unique_books = df_interactions[
    ['item_id', 'item_series_id']
].drop_duplicates()

resolved_dates  = {}
removed_no_year = 0

# Series books — check slot conflicts within each series
series_books_df = unique_books[
    unique_books['item_series_id'].str.startswith('Series_')
].copy()
standalone_df = unique_books[
    unique_books['item_series_id'].str.startswith('SB')
].copy()

for series_id, group in series_books_df.groupby('item_series_id'):
    bids = group['item_id'].tolist()
    # highest rated picks slot first
    bids_sorted = sorted(
        bids,
        key=lambda b: book_meta_raw.get(b, {}).get('ratings_count', 0),
        reverse=True
    )
    used_slots = set()
    for bid in bids_sorted:
        slot = resolve_date(bid, used_slots)
        if slot is None:
            removed_no_year += 1
        else:
            resolved_dates[bid] = slot
            used_slots.add(slot)

# Standalone books — no slot conflicts
for bid in standalone_df['item_id'].tolist():
    slot = resolve_date(bid, set())
    if slot is None:
        removed_no_year += 1
    else:
        resolved_dates[bid] = slot

print(f"  Books removed (no pub_year): {removed_no_year:,}")
print(f"  Books with resolved dates:   {len(resolved_dates):,}")

# ── STEP 7: Assign sequence positions ────────────────────────────
print(" assigning sequence positions")

date_records = []
for bid, (yr, mo, dy) in resolved_dates.items():
    date_records.append({
        'item_id':       bid,
        'pub_year':      yr,
        'pub_month':     mo,
        'pub_day':       dy,
        'ratings_count': book_meta_raw.get(
            bid, {}
        ).get('ratings_count', 0)
    })
metadata_df = pd.DataFrame(date_records)

unique_books_clean = unique_books[
    unique_books['item_id'].isin(resolved_dates)
].copy()
merged_df = unique_books_clean.merge(metadata_df, on='item_id', how='left')

series_mask     = merged_df['item_series_id'].str.startswith('Series_')
standalone_mask = merged_df['item_series_id'].str.startswith('SB')

series_df    = merged_df[series_mask].copy()
standalone_s = merged_df[standalone_mask].copy()

# title position to series_df
series_df['title_pos'] = series_df['item_id'].map(
    lambda b: book_title_pos.get(b)
)

# PRIMARY: use title position if available
# FALLBACK: use publication date
series_df['has_title_pos'] = series_df['title_pos'].notna()

# Sort: items WITH title position first by title_pos
# items WITHOUT title position by pub_year/month/day
series_df = series_df.sort_values(
    by=['item_series_id', 'has_title_pos',
        'title_pos',
        'pub_year', 'pub_month', 'pub_day',
        'ratings_count'],
    ascending=[True, False, True,
               True, True, True, False]
)
series_df['position_in_series'] = (
    series_df.groupby('item_series_id').cumcount() + 1
)
series_df = series_df.drop(columns=['title_pos', 'has_title_pos'])
standalone_s['position_in_series'] = 1

pos_table = pd.concat(
    [series_df, standalone_s], ignore_index=True
)[['item_id', 'item_series_id', 'position_in_series']]

print(f"  Series books: {len(series_df):,}")
print(f"  Standalone books: {len(standalone_s):,}")
print(" building final output")

# Remove interactions for books with no pub_year
df_interactions_clean = df_interactions[
    df_interactions['item_id'].isin(resolved_dates)
].copy()

removed_interactions = len(df_interactions) - len(df_interactions_clean)
print(f"  Interactions removed (no pub_year): {removed_interactions:,}")

df_final = df_interactions_clean.merge(
    pos_table, on=['item_id', 'item_series_id'], how='left'
)

def build_series_id(row):
    sid = str(row['item_series_id'])
    pos = row['position_in_series']
    if sid.startswith('SB'):
        return sid
    return f"{sid}_P{int(pos)}"

df_final['item_series_id'] = df_final.apply(build_series_id, axis=1)
output_df = df_final[
    ['user_id', 'item_id', 'time', 'item_series_id']
].copy()
output_df.to_csv(FINAL_CSV, index=False)

size_mb = os.path.getsize(FINAL_CSV) / 1024 / 1024
print(f"\nSaved: {FINAL_CSV}  ({size_mb:.1f} MB)")

series_rows = output_df[
    output_df['item_series_id'].str.startswith('Series_')
]
sb_rows = output_df[
    output_df['item_series_id'].str.startswith('SB')
]
old_format = series_rows[
    ~series_rows['item_series_id'].str.contains('_P')
]

print(f"\nSummary — {GENRE_LABEL}")
print(f"  rows        : {len(output_df):,}")
print(f"  columns     : {list(output_df.columns)}")
print(f"  user range  : 0 to {output_df['user_id'].max():,}")
print(f"  ISQ rows    : {len(series_rows):,}")
print(f"  ISA rows    : {len(sb_rows):,}")
print(f"  old format  : {len(old_format):,}  (should be 0)")
print(f"\nFirst 3 rows:")
print(output_df.head(3).to_string(index=False))
print(f"\nLast 3 rows:")
print(output_df.tail(3).to_string(index=False))


# ===== CELL 44 =====
# ─────────────────────────────────────────────────────────────────

# Creates smaller test versions of all 4 genres
# Output: Goodreads_{GENRE}_HSAL_100k.csv for each genre
# ─────────────────────────────────────────────────────────────────
import pandas as pd
import random
import os

random.seed(42)

SAVE_DIR = '/content/drive/MyDrive/HSAL_Goodreads/'

GENRES = {
    'mystery':            'Goodreads_Mystery_HSAL_Final.csv',
    'fantasy_paranormal': 'Goodreads_Fantasy_HSAL_Final.csv',
    'young_adult':        'Goodreads_YoungAdult_HSAL_Final.csv',
    'romance':            'Goodreads_Romance_HSAL_Final.csv',
}

TARGET_ROWS = 100_000

print("Creating 100k test datasets for all 4 genres")
print("=" * 60)

for genre, source_file in GENRES.items():
    source_path = SAVE_DIR + source_file
    output_path = SAVE_DIR + f'Goodreads_{genre}_HSAL_100k.csv'

    if not os.path.exists(source_path):
        print(f"\n⚠ SKIPPING {genre} — source file not found: {source_file}")
        continue

    df = pd.read_csv(source_path)
    total_rows = len(df)

    if total_rows <= TARGET_ROWS:
        print(f"\n{genre}: only {total_rows:,} rows — copying as is")
        df.to_csv(output_path, index=False)
        continue


    import math
    from collections import defaultdict

    # Extract base series ID (strip _P suffix)
    def base_series(sid):
        if str(sid).startswith('SB'):
            return sid
        parts = str(sid).rsplit('_P', 1)
        return parts[0] if len(parts) == 2 else sid

    df['_base_series'] = df['item_series_id'].apply(base_series)

    # Count rows per series
    series_counts = df['_base_series'].value_counts().to_dict()

    # Compute sqrt-weighted budget per series
    weights  = {s: math.sqrt(c) for s, c in series_counts.items()}
    total_w  = sum(weights.values())
    budgets  = {
        s: max(1, round(w / total_w * TARGET_ROWS))
        for s, w in weights.items()
    }

    # Adjust to hit exactly TARGET_ROWS
    diff = TARGET_ROWS - sum(budgets.values())
    top  = sorted(budgets, key=lambda k: weights[k], reverse=True)
    for i in range(abs(diff)):
        budgets[top[i % len(top)]] += int(math.copysign(1, diff))

    # Sample from each series
    sampled = []
    for series_id, group in df.groupby('_base_series'):
        budget = budgets.get(series_id, 1)
        if len(group) <= budget:
            sampled.append(group)
        else:
            sampled.append(group.sample(n=budget, random_state=42))

    df_100k = pd.concat(sampled, ignore_index=True)
    df_100k = df_100k.drop(columns=['_base_series'])

    # Shuffle rows
    df_100k = df_100k.sample(frac=1, random_state=42).reset_index(drop=True)
    df_100k.to_csv(output_path, index=False)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    series_rows = df_100k[df_100k['item_series_id'].str.startswith('Series_')]
    sb_rows     = df_100k[df_100k['item_series_id'].str.startswith('SB')]

    print(f"\n{genre}:")
    print(f"  Source rows:    {total_rows:,}")
    print(f"  Output rows:    {len(df_100k):,}  (target {TARGET_ROWS:,})")
    print(f"  Unique items:   {df_100k['item_id'].nunique():,}")
    print(f"  Unique users:   {df_100k['user_id'].nunique():,}")
    print(f"  ISQ rows:       {len(series_rows):,}")
    print(f"  ISA rows:       {len(sb_rows):,}")
    print(f"  File size:      {size_mb:.1f} MB")
    print(f"  Saved:          {output_path}")

print("\n" + "=" * 60)
print("Done. Original files untouched.")
print("New test files saved as Goodreads_{genre}_HSAL_100k.csv")
