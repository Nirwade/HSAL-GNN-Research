"""
Builds prequel/sequel ground truth for the FULL corpus, directly from the
r_sequel column already present in Goodreads_young_adult_item_meta_with_title.csv.

r_sequel semantics (verified against real rows, not assumed):
  - r_sequel lists ALL later books in the same series known to this file,
    not just the immediate next volume (e.g. Book of Ember #1 lists both
    #3 and #4). So this is a valid signal for a binary "has a sequel /
    has a prequel" label, but a given book can have more than one entry
    in prequel_item_ids / sequel_item_ids — that's expected, not a bug.

Labels produced (booleans, per item_id):
  ground_truth_has_sequel  = r_sequel list is non-empty for this item
  ground_truth_has_prequel = this item_id appears in >=1 OTHER item's r_sequel list

No manual labeling, no sampling — every one of the 8,108 books gets a label.
Does NOT cover alternative_version: there is no structural field for that
relation anywhere in this dataset, so no ground truth can be derived this way.
That relation still needs manual verification if you want accuracy numbers
on the full set.
"""

import ast
from collections import defaultdict

import pandas as pd

INPUT_CSV = "Goodreads_young_adult_item_meta_with_title.csv"
OUTPUT_CSV = "ground_truth_full_corpus.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    df["r_sequel_list"] = df["r_sequel"].apply(ast.literal_eval)

    # has_sequel: direct from the column
    df["ground_truth_has_sequel"] = df["r_sequel_list"].apply(lambda lst: len(lst) > 0)
    df["sequel_item_ids"] = df["r_sequel_list"].apply(lambda lst: lst)

    # has_prequel: invert the relation — build a map of target_id -> [source_ids]
    prequel_of = defaultdict(list)
    for _, row in df.iterrows():
        for target_id in row["r_sequel_list"]:
            prequel_of[target_id].append(row["item_id"])

    df["prequel_item_ids"] = df["item_id"].apply(lambda iid: prequel_of.get(iid, []))
    df["ground_truth_has_prequel"] = df["prequel_item_ids"].apply(lambda lst: len(lst) > 0)

    out = df[[
        "item_id", "title",
        "ground_truth_has_prequel", "prequel_item_ids",
        "ground_truth_has_sequel", "sequel_item_ids",
    ]]
    out.to_csv(OUTPUT_CSV, index=False)

    n = len(out)
    print(f"Total items: {n:,}")
    print(f"has_prequel=True: {out['ground_truth_has_prequel'].sum():,} "
          f"({100*out['ground_truth_has_prequel'].mean():.1f}%)")
    print(f"has_sequel=True:  {out['ground_truth_has_sequel'].sum():,} "
          f"({100*out['ground_truth_has_sequel'].mean():.1f}%)")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
