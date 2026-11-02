"""
Prequel candidate generation via local LLMs (Ollama) for HSAL-GNN.

Title-hallucination flag: for every True result, ask a separate isolated
existence-check on the returned prequel title/author and record it as a
soft signal (title_check column) — does not overwrite the row. The
false-negative recheck mechanism was tested across three prompt designs
and dropped: it never once corrected a false negative in ~85 attempts,
likely because a fresh, stateless API call has no real continuity with
the model's own prior reasoning to enforce consistency.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
CHECKPOINT_EVERY = 25
REQUEST_TIMEOUT = 60
MAX_RETRIES = 2

FILES = {
    "A": "Goodreads_young_adult_item_meta_with_title.csv",
    "B": "Goodreads_young_adult_author.meta.csv",
}
EVAL_FILE = "prequel_sample_400.csv"

PROMPT_NO_AUTHOR = """You are a book-knowledge assistant. You are given the title of a book. \
Using your own knowledge of this book's universe, its series, and its fictional universe, determine \
whether it has a PREQUEL — a book set chronologically or narratively before it, in the \
same series or universe.

Book title: "{title}"

Think through the evidence first, then give your answer. Respond with ONLY a JSON object, \
no other text, in exactly this format:
{{
  "series_info": "series name and position if known, or empty string if unknown",
  "reason": "step-by-step: does this book's series/universe have an earlier book? name it if so, or state clearly why not",
  "confidence": "high" or "medium" or "low",
  "has_prequel": true or false,
  "prequel_title": "exact title of the prequel, or empty string if none",
  "prequel_author": "author name, or empty string if none",
  "prequel_description": "1-2 sentence plot summary of the prequel, or empty string if none"
}}

If you are not confident a prequel exists, or the book is standalone, return has_prequel: false \
with empty strings for the other fields. Do not invent a prequel that does not exist."""


PROMPT_WITH_AUTHOR = """You are a book-knowledge assistant. You are given the title and author of a book. \
Using your own knowledge of this book's universe, its series, and its fictional universe, determine \
whether it has a PREQUEL — a book set chronologically or narratively before it, in the \
same series or universe.

Book title: "{title}"
Author: "{author}"

Think through the evidence first, then give your answer. Respond with ONLY a JSON object, \
no other text, in exactly this format:
{{
  "series_info": "series name and position if known, or empty string if unknown",
  "reason": "step-by-step: does this book's series/universe have an earlier book? name it if so, or state clearly why not",
  "confidence": "high" or "medium" or "low",
  "has_prequel": true or false,
  "prequel_title": "exact title of the prequel, or empty string if none",
  "prequel_author": "author name, or empty string if none",
  "prequel_description": "1-2 sentence plot summary of the prequel, or empty string if none"
}}

If you are not confident a prequel exists, or the book is standalone, return has_prequel: false \
with empty strings for the other fields. Do not invent a prequel that does not exist."""


def _chat_json(model, prompt_text, timeout=REQUEST_TIMEOUT):
    """Low-level: send one prompt, get back parsed JSON dict or None."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt_text}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.2},
    }
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["message"]["content"]
            return json.loads(content), None
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            if attempt < MAX_RETRIES:
                time.sleep(2)
                continue
            return None, type(e).__name__


def call_ollama(model, title, author, prompt_template):
    kwargs = {"title": title}
    if author is not None:
        kwargs["author"] = author or "unknown"
    start = time.time()
    parsed, err = _chat_json(model, prompt_template.format(**kwargs))
    elapsed = time.time() - start

    if parsed is None:
        return {
            "has_prequel": False, "prequel_title": "", "prequel_author": "",
            "prequel_description": "", "series_info": "", "confidence": "",
            "reason": "", "parse_status": f"fail:{err}", "call_seconds": round(elapsed, 2),
        }

    result = {
        "has_prequel": bool(parsed.get("has_prequel", False)),
        "prequel_title": str(parsed.get("prequel_title", "")).strip(),
        "prequel_author": str(parsed.get("prequel_author", "")).strip(),
        "prequel_description": str(parsed.get("prequel_description", "")).strip(),
        "series_info": str(parsed.get("series_info", "")).strip(),
        "confidence": str(parsed.get("confidence", "")).strip(),
        "reason": str(parsed.get("reason", "")).strip(),
        "parse_status": "ok",
        "call_seconds": round(elapsed, 2),
    }
    # Step 3 fix: a True verdict with no actual title is unusable — force it False
    if result["has_prequel"] and not result["prequel_title"]:
        result["has_prequel"] = False
        result["parse_status"] = "corrected:empty_title"
    return result


def needs_title_check(result):
    # Unconditional now: confidence/hedge language proved unreliable as a gate —
    # a model can hallucinate fluently at "high" confidence with no hedge words at all.
    return bool(result["has_prequel"] and result["prequel_title"])


def verify_title_exists(model, prequel_title, prequel_author):
    author_clause = f" by {prequel_author}" if prequel_author else ""
    prompt = f"""Does a real, published book titled "{prequel_title}"{author_clause} actually exist? \
Answer based only on what you are confident is real, not a guess. Respond with ONLY JSON:
{{"exists": true or false, "confidence": "high" or "medium" or "low"}}"""
    parsed, err = _chat_json(model, prompt)
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
    args = parser.parse_args()

    if args.eval:
        input_csv = EVAL_FILE
        has_author = False
        prompt_template = PROMPT_NO_AUTHOR
    else:
        input_csv = FILES[args.version]
        has_author = args.version == "B"
        prompt_template = PROMPT_WITH_AUTHOR if has_author else PROMPT_NO_AUTHOR

    if not os.path.exists(input_csv):
        print(f"ERROR: input file not found: {input_csv}")
        sys.exit(1)

    books = []
    with open(input_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            books.append({
                "item_id": row["item_id"],
                "title": row["title"],
                "author": row.get("authors", "") if has_author else None,
            })
    if args.limit:
        books = books[: args.limit]

    label = "SAMPLE400" if args.eval else f"Version {args.version}"
    print(f"{label}   Model: {args.model}   Books: {len(books):,}")

    output_dir = "prequel_extraction"
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_test" if args.limit else ""
    prefix = "sample400" if args.eval else f"v{args.version}"
    output_path = os.path.join(
        output_dir, f"prequels_{prefix}_{args.model.replace(':', '_')}{suffix}_verified.csv"
    )

    done_ids = load_done_ids(output_path)
    file_exists = os.path.exists(output_path) and len(done_ids) > 0
    remaining = [b for b in books if b["item_id"] not in done_ids]

    print(f"Output: {output_path}")
    print(f"Already done (resume): {len(done_ids):,}   Remaining: {len(remaining):,}")
    if not remaining:
        print("Nothing to do.")
        return

    fieldnames = [
        "item_id", "title", "author", "has_prequel",
        "prequel_title", "prequel_author", "prequel_description",
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
            result = call_ollama(args.model, book["title"], book["author"], prompt_template)
            title_check = ""

            # Title-hallucination flag (soft signal only, doesn't change has_prequel)
            if needs_title_check(result):
                title_check = verify_title_exists(
                    args.model, result["prequel_title"], result["prequel_author"]
                )

            if result["parse_status"].startswith("fail"):
                stats["parse_fail"] += 1
            elif result["has_prequel"]:
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
                print(
                    f"  [{i}/{len(remaining)}] yes={stats['true']} no={stats['false']} "
                    f"fail={stats['parse_fail']} avg={avg:.2f}s/call "
                    f"ETA={eta_min:.1f}min @ {datetime.now().strftime('%H:%M:%S')}"
                )

    total = time.time() - model_start
    print(f"\nDone. yes={stats['true']} no={stats['false']} fail={stats['parse_fail']}")
    print(f"Total: {total/60:.1f} min for {len(remaining)} books ({total/max(len(remaining),1):.2f}s/call avg)")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()