"""Freeze a book-balanced exploratory sample before generating predictions."""

from collections import defaultdict
import argparse
import csv
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import random
import re

from src.guofeng import read_tagged


def chapters(path):
    key, lines = None, []
    for event in read_tagged(path, allow_blank=True):
        if event[0] == "text":
            key = event[1]
            # Repeated inline markers observed in both languages are artifacts.
            # Apply the same fixed cleanup before sampling, context, and scoring.
            lines.append(event[2].replace("#<#", "").replace("#>#", "").strip())
        elif event[1] == "</CHAPTER>":
            if key is None:
                raise ValueError("Empty chapter")
            yield key, lines
            key, lines = None, []


def eligible_positions(source, reference, require_content=False):
    """Exclude heading and its use as context; require 3 prior body rows."""
    if len(source) != len(reference) or any(not s.strip() for s in source):
        return []
    return [i for i in range(4, len(source)) if reference[i].strip()
            and (not require_content or re.search(r"[\u3400-\u9fffA-Za-z0-9]", source[i]))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--books", type=int, default=10)
    parser.add_argument("--chapters-per-book", type=int, default=2)
    parser.add_argument("--targets-per-chapter", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--prefix", default="study")
    parser.add_argument("--require-content", action="store_true")
    args = parser.parse_args()
    if min(args.books, args.chapters_per_book, args.targets_per_chapter) < 1:
        parser.error("Sample dimensions must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.prefix):
        parser.error("prefix must be a simple filename prefix")
    output = Path(f"data/processed/{args.prefix}_train.csv")
    manifest = Path(f"data/processed/{args.prefix}_targets.json")
    if output.exists() or manifest.exists():
        parser.error("Frozen study sample already exists")
    base = Path("GuoFeng_Webnovel_Corpus/V1/TRAIN")
    source_path, reference_path = base / "train.zh", base / "train.en"
    rng = random.Random(args.seed)
    reservoirs, counts = defaultdict(list), defaultdict(int)
    rejected = []
    total = 0
    # Reservoir sampling stores only two chapters per book, rather than the corpus.
    for source, reference in zip_longest(chapters(source_path), chapters(reference_path)):
        if source is None or reference is None or source[0] != reference[0]:
            raise ValueError("Chapter IDs/order differ; no sample written")
        key, zh = source
        en = reference[1]
        total += 1
        positions = eligible_positions(zh, en, args.require_content)
        if len(positions) < args.targets_per_chapter:
            rejected.append({"book_id": key[0], "chapter_id": key[1],
                             "source_rows": len(zh), "reference_rows": len(en)})
            continue
        counts[key[0]] += 1
        item = (key, zh, en, positions)
        pool = reservoirs[key[0]]
        if len(pool) < args.chapters_per_book:
            pool.append(item)
        else:
            slot = rng.randrange(counts[key[0]])
            if slot < args.chapters_per_book:
                pool[slot] = item
    eligible_books = sorted(b for b, pool in reservoirs.items() if len(pool) == args.chapters_per_book)
    if len(eligible_books) < args.books:
        raise ValueError("Insufficient eligible novels")
    books = sorted(rng.sample(eligible_books, args.books))
    targets = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chapter_id", "sentence_id", "source", "reference"])
        for book in books:
            for key, zh, en, positions in sorted(reservoirs[book], key=lambda item: item[0]):
                chapter_id = f"{len(book)}:{book}:{key[1]}"
                selected = sorted(rng.sample(positions, args.targets_per_chapter))
                targets.extend({"chapter_id": chapter_id, "sentence_id": i,
                                "book_id": book} for i in selected)
                # Retain every body row so sampled targets keep their true history.
                writer.writerows((chapter_id, i, zh[i], en[i]) for i in range(1, len(zh)))
    metadata = dict(seed=args.seed, split="TRAIN", books=books, chapters=args.books * args.chapters_per_book,
                    require_content=args.require_content,
                    dimensions=dict(books=args.books, chapters_per_book=args.chapters_per_book,
                                    targets_per_chapter=args.targets_per_chapter),
                    targets=targets, target_count=len(targets),
                    policy=f"{args.books} books uniformly; {args.chapters_per_book} eligible chapters/book by reservoir; {args.targets_per_chapter} targets/chapter uniformly; exclude first content row as heading; require 3 body predecessors; references never used for sampling except nonempty check; content filter={args.require_content}",
                    cleanup="Remove exact #<# and #># markers in both languages; strip outer whitespace; raw files unchanged",
                    raw_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [source_path, reference_path]},
                    dataset_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                    chapters_scanned=total, eligible_books=len(counts), rejected_chapters=rejected)
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(f"Frozen {len(targets)} targets, {metadata['chapters']} chapters, {len(books)} books; rejected {len(rejected)} chapters")


if __name__ == "__main__":
    main()
