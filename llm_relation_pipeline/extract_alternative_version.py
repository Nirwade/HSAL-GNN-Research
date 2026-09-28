"""
Alternative-version extraction driver. Relation meaning (confirmed by Shashi,
not the "different edition/format" reading): an alternate-universe, alternate-
timeline, "what if", or divergent-canon version of the SAME basic story —
same core premise/characters, but a fundamentally different version of events,
not a chronological sequel/prequel and not a different author's retelling.

No positional meaning here — matching_core.py's series-position rerank does
NOT apply to this relation type; only exact match + cosine + coherence.
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime

from ollama_client import chat_json

FILES = {
    "A": "Goodreads_young_adult_item_meta_with_title.csv",
    "B": "Goodreads_young_adult_author.meta.csv",
}
EVAL_FILE = "alternative_version_eval_set_29.csv"
CHECKPOINT_EVERY = 25

PROMPT_NO_AUTHOR = """You are a book-knowledge assistant. You are given the title of a book. \
Using your own knowledge, determine whether this book has an ALTERNATIVE VERSION.

DEFINITION: Alternative Version means a materially different narrative version of an existing \
story, characters, or core narrative premise. This includes:
- An alternate timeline where a major event has a different outcome.
- An alternate-universe version using the same characters or core premise.
- An explicit "what if" version of the original story.
- A substantial graphic novel or manga adaptation of the same novel.
- A movie, TV, or anime adaptation that presents the same underlying narrative as a distinct work.
- A deliberately altered alternate-canon version of the original story.

The relationship MUST come from the underlying narrative being changed, branched, or substantially \
adapted. The following do NOT qualify, even if they seem related:
- A normal sequel or prequel.
- A companion novel in the same continuous canon.
- A spin-off or side story.
- A simple jump forward or backward in time (that's still the same canon, not an alternative version).
- A different cover or ordinary illustrated edition.
- An abridged/unabridged edition without narrative changes.
- A book that only references or shares characters with another work.
- Same title belonging to an unrelated author or story (a title collision is NOT evidence of a relationship).

Book title: "{title}"

Think through the evidence first, then give your answer. Respond with ONLY a JSON object, \
no other text, in exactly this format:
{{
  "series_info": "series name and position if known, or empty string if unknown",
  "reason": "step-by-step: is there a materially different narrative version (alt-timeline/alt-canon/adaptation)? If you name one, explain specifically HOW the narrative differs, not just that a related book/adaptation exists. If none, state why the candidates you considered don't qualify (e.g. 'X is a companion novel in the same canon, not a narrative branch').",
  "confidence": "high" or "medium" or "low",
  "has_alternative_version": true or false,
  "match_title": "exact title of the alternative version, or empty string if none",
  "match_author": "author/studio/creator name — leave this EMPTY if you are not certain, do not guess an author for a title you're unsure about",
  "match_description": "1-2 sentences on specifically how the narrative diverges, or empty string if none",
  "match_medium": "book, movie, tv, or anime — whichever the alternative version actually is, or empty string if has_alternative_version is false"
}}

This is a genuinely rare relationship — most books do NOT have one, and it is NOT the same as having \
a sequel or companion novel. If you cannot name a specific alternative version where the narrative \
itself is materially different, return has_alternative_version: false with empty strings for the \
other fields. Do not invent one, and do not guess an author you aren't sure of."""

PROMPT_WITH_AUTHOR = """You are a book-knowledge assistant. You are given the title and author of a book. \
Using your own knowledge, determine whether this book has an ALTERNATIVE VERSION.

DEFINITION: Alternative Version means a materially different narrative version of an existing \
story, characters, or core narrative premise. This includes:
- An alternate timeline where a major event has a different outcome.
- An alternate-universe version using the same characters or core premise.
- An explicit "what if" version of the original story.
- A substantial graphic novel or manga adaptation of the same novel.
- A movie, TV, or anime adaptation that presents the same underlying narrative as a distinct work.
- A deliberately altered alternate-canon version of the original story.

The relationship MUST come from the underlying narrative being changed, branched, or substantially \
adapted. The following do NOT qualify, even if they seem related:
- A normal sequel or prequel.
- A companion novel in the same continuous canon.
- A spin-off or side story.
- A simple jump forward or backward in time (that's still the same canon, not an alternative version).
- A different cover or ordinary illustrated edition.
- An abridged/unabridged edition without narrative changes.
- A book that only references or shares characters with another work.
- Same title belonging to an unrelated author or story (a title collision is NOT evidence of a relationship).

Book title: "{title}"
Author: "{author}"

Think through the evidence first, then give your answer. Respond with ONLY a JSON object, \
no other text, in exactly this format:
{{
  "series_info": "series name and position if known, or empty string if unknown",
  "reason": "step-by-step: is there a materially different narrative version (alt-timeline/alt-canon/adaptation)? If you name one, explain specifically HOW the narrative differs, not just that a related book/adaptation exists. If none, state why the candidates you considered don't qualify (e.g. 'X is a companion novel in the same canon, not a narrative branch').",
  "confidence": "high" or "medium" or "low",
  "has_alternative_version": true or false,
  "match_title": "exact title of the alternative version, or empty string if none",
  "match_author": "author/studio/creator name — leave this EMPTY if you are not certain, do not guess an author for a title you're unsure about",
  "match_description": "1-2 sentences on specifically how the narrative diverges, or empty string if none",
  "match_medium": "book, movie, tv, or anime — whichever the alternative version actually is, or empty string if has_alternative_version is false"
}}

This is a genuinely rare relationship — most books do NOT have one, and it is NOT the same as having \
a sequel or companion novel. If you cannot name a specific alternative version where the narrative \
itself is materially different, return has_alternative_version: false with empty strings for the \
other fields. Do not invent one, and do not guess an author you aren't sure of."""


def call_model(model, title, author, prompt_template):
    kwargs = {"title": title}
    if author is not None:
        kwargs["author"] = author or "unknown"
    start = time.time()
    parsed, err = chat_json(model, prompt_template.format(**kwargs))
    elapsed = time.time() - start

    if parsed is None:
        return {
            "has_alternative_version": False, "match_title": "", "match_author": "",
            "match_description": "", "match_medium": "", "series_info": "", "confidence": "",
            "reason": "", "parse_status": f"fail:{err}", "call_seconds": round(elapsed, 2),
        }

    result = {
        "has_alternative_version": bool(parsed.get("has_alternative_version", False)),
        "match_title": str(parsed.get("match_title", "")).strip(),
        "match_author": str(parsed.get("match_author", "")).strip(),
        "match_description": str(parsed.get("match_description", "")).strip(),
        "match_medium": str(parsed.get("match_medium", "")).strip().lower(),
        "series_info": str(parsed.get("series_info", "")).strip(),
        "confidence": str(parsed.get("confidence", "")).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
        "parse_status": "ok",
        "call_seconds": round(elapsed, 2),
    }
    if result["has_alternative_version"] and not result["match_title"]:
        result["has_alternative_version"] = False
        result["parse_status"] = "corrected:empty_title"

    # Code-level rule: if the model's own reasoning names the excluded
    # sequel/prequel categories, it's leaking that relationship regardless
    # of what it marked has_alternative_version as — force it False rather
    # than trust a boolean that contradicts its own stated reasoning.
    # Checks BOTH reason and match_description: the model inconsistently
    # puts its actual reasoning in either field (sometimes leaving "reason"
    # empty and explaining itself inside match_description instead), so a
    # single-field check misses real leaks.
    combined_text = f"{result['reason']} {result['match_description']}".lower()
    if result["has_alternative_version"] and ("sequel" in combined_text or "prequel" in combined_text):
        result["has_alternative_version"] = False
        result["parse_status"] = "corrected:sequel_prequel_leak"

    return result


def needs_title_check(result):
    return bool(result["has_alternative_version"] and result["match_title"])


def verify_title_exists(model, match_title, match_author, match_medium):
    author_clause = f" by {match_author}" if match_author else ""
    medium_word = {"movie": "movie", "tv": "TV show", "anime": "anime"}.get(match_medium, "book")
    prompt = f"""Does a real {medium_word} titled "{match_title}"{author_clause} actually exist? \
Answer based only on what you are confident is real, not a guess. Respond with ONLY JSON:
{{"exists": true or false, "confidence": "high" or "medium" or "low"}}"""
    parsed, err = chat_json(model, prompt)
    if parsed is None:
        return "uncertain"
    exists = bool(parsed.get("exists", False))
    conf = str(parsed.get("confidence", "")).strip().lower()
    if exists and conf == "high":
        return "verified"
    if not exists:
        return "not_verified"
    return "uncertain"


def load_done_ids(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["item_id"])
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", choices=["A", "B"], default="A")
    parser.add_argument("--model", choices=["llama3.1:8b", "mistral"], required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--blind", action="store_true",
                         help="Run a random N-book blind test (no ground truth needed) via --limit")
    args = parser.parse_args()

    if args.eval:
        input_csv = EVAL_FILE
    else:
        input_csv = FILES[args.version]

    has_author = args.version == "B"
    prompt_template = PROMPT_WITH_AUTHOR if has_author else PROMPT_NO_AUTHOR

    if not os.path.exists(input_csv):
        print(f"ERROR: input file not found: {input_csv}")
        sys.exit(1)

    author_lookup = {}
    if has_author:
        with open(FILES["B"], "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                author_lookup[row["item_id"]] = row.get("authors", "")

    books = []
    with open(input_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            books.append({
                "item_id": row["item_id"],
                "title": row["title"],
                "author": author_lookup.get(row["item_id"], "") if has_author else None,
            })
    if args.limit:
        books = books[: args.limit]

    label = f"EVAL (Version {args.version})" if args.eval else f"Version {args.version}"
    print(f"[alternative_version] {label}   Model: {args.model}   Books: {len(books):,}")

    output_dir = "alternative_version_extraction"
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_test" if args.limit else ""
    if args.eval:
        prefix = "eval" if args.version == "A" else f"eval_v{args.version}"
    else:
        prefix = f"v{args.version}"
    output_path = os.path.join(output_dir, f"altver_{prefix}_{args.model.replace(':', '_')}{suffix}_verified.csv")

    done_ids = load_done_ids(output_path)
    file_exists = os.path.exists(output_path) and len(done_ids) > 0
    remaining = [b for b in books if b["item_id"] not in done_ids]

    print(f"Output: {output_path}")
    print(f"Already done (resume): {len(done_ids):,}   Remaining: {len(remaining):,}")
    if not remaining:
        print("Nothing to do.")
        return

    fieldnames = [
        "item_id", "title", "author", "has_alternative_version",
        "match_title", "match_author", "match_description", "match_medium",
        "series_info", "confidence", "reason",
        "parse_status", "call_seconds", "title_check",
    ]

    stats = {"true": 0, "false": 0, "parse_fail": 0}
    model_start = time.time()

    mode = "a" if file_exists else "w"
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, book in enumerate(remaining, 1):
            result = call_model(args.model, book["title"], book["author"], prompt_template)
            title_check = ""
            if needs_title_check(result):
                title_check = verify_title_exists(
                    args.model, result["match_title"], result["match_author"], result["match_medium"]
                )

            if result["parse_status"].startswith("fail"):
                stats["parse_fail"] += 1
            elif result["has_alternative_version"]:
                stats["true"] += 1
            else:
                stats["false"] += 1

            writer.writerow({
                "item_id": book["item_id"],
                "title": book["title"],
                "author": book["author"] or "",
                **result,
                "title_check": title_check,
            })
            f.flush()

            if i % CHECKPOINT_EVERY == 0 or i == len(remaining):
                elapsed_total = time.time() - model_start
                avg = elapsed_total / i
                eta_min = (len(remaining) - i) * avg / 60
                print(f"  [{i}/{len(remaining)}] yes={stats['true']} no={stats['false']} "
                      f"fail={stats['parse_fail']} avg={avg:.2f}s/call ETA={eta_min:.1f}min "
                      f"@ {datetime.now().strftime('%H:%M:%S')}")

    total = time.time() - model_start
    print(f"\nDone. yes={stats['true']} no={stats['false']} fail={stats['parse_fail']}")
    print(f"Total: {total/60:.1f} min for {len(remaining)} books ({total/max(len(remaining),1):.2f}s/call avg)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()