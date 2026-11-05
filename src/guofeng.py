"""Import the tagged Chinese–English V1 format documented upstream.

Use line pairing, never independent sentence splitting. Fail closed if files
do not have identical chapter structure and aligned content lines. Missing
references retain their source position for context but cannot be scored.
"""

import argparse
import csv
from itertools import zip_longest
from pathlib import Path
import re

from src.dataset import Sentence

OPEN_TAG = re.compile(r'<(BOOK|CHAPTER)\s+id="([^"]+)"\s*>')


def read_tagged(path: Path, allow_blank: bool = False):
    book = chapter = None
    seen = set()
    with path.open(encoding="utf-8-sig") as handle:
        for number, raw in enumerate(handle, 1):
            text = raw.strip()
            if not text:
                if chapter is not None:
                    if not allow_blank:
                        raise ValueError(f"{path}:{number}: blank source line")
                    yield ("text", (book, chapter), "")
                continue
            opening = OPEN_TAG.fullmatch(text)
            if opening:
                kind, identifier = opening.groups()
                if kind == "BOOK":
                    if book is not None:
                        raise ValueError(f"{path}:{number}: nested BOOK")
                    book = identifier
                else:
                    if book is None or chapter is not None:
                        raise ValueError(f"{path}:{number}: invalid CHAPTER nesting")
                    chapter = identifier
                    key = (book, chapter)
                    if key in seen:
                        raise ValueError(f"{path}:{number}: duplicate chapter")
                    seen.add(key)
                yield ("tag", text)
            elif text in ("</CHAPTER>", "</BOOK>"):
                if text == "</CHAPTER>":
                    if chapter is None:
                        raise ValueError(f"{path}:{number}: unmatched chapter close")
                    chapter = None
                else:
                    if book is None or chapter is not None:
                        raise ValueError(f"{path}:{number}: invalid book close")
                    book = None
                yield ("tag", text)
            else:
                if text.startswith(("<BOOK", "<CHAPTER", "</BOOK", "</CHAPTER")):
                    raise ValueError(f"{path}:{number}: malformed structural tag")
                if chapter is None:
                    raise ValueError(f"{path}:{number}: text outside chapter")
                yield ("text", (book, chapter), text)
    if book is not None or chapter is not None:
        raise ValueError(f"{path}: unclosed structural tags")


def load_guofeng(source_path: Path, reference_path: Path) -> list[Sentence]:
    sentences = []
    current = None
    index = 0
    for source, reference in zip_longest(
        read_tagged(source_path), read_tagged(reference_path, allow_blank=True)
    ):
        if source is None or reference is None or source[:2] != reference[:2]:
            raise ValueError("Source/reference structure or aligned line counts differ")
        if source[0] == "tag":
            continue
        key = source[1]
        if key != current:
            current, index = key, 0
        # Length prefixes avoid collisions even when IDs contain separators.
        book, chapter = key
        chapter_id = f"{len(book)}:{book}:{chapter}"
        sentences.append(Sentence(chapter_id, index, source[2], reference[2]))
        index += 1
    if not sentences:
        raise ValueError("No aligned sentences found")
    return sentences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/processed/guofeng.csv"))
    args = parser.parse_args()
    if args.output.resolve() in {args.source.resolve(), args.reference.resolve()}:
        parser.error("output must not overwrite an input file")
    sentences = load_guofeng(args.source, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["chapter_id", "sentence_id", "source", "reference"])
        for row in sentences:
            writer.writerow([row.chapter_id, row.sentence_id, row.source, row.reference])
    print(f"Imported {len(sentences)} aligned rows to {args.output}")


if __name__ == "__main__":
    main()
