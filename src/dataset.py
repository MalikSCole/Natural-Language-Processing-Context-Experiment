"""Load aligned sentences from a normalized UTF-8 CSV.

The documented GuoFeng V1 adapter lives in src.guofeng. This normalized
loader deliberately does not guess how raw corpus files are aligned.
"""

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Sentence:
    chapter_id: str
    sentence_id: int
    source: str
    reference: str


def load_dataset(path: str | Path, allow_missing_references: bool = False) -> list[Sentence]:
    """Validate and load rows, preserving chapter and sentence file order.

    IDs must increase within each chapter. Chapter IDs must be globally
    unique (include the book ID if necessary), and chapters must be contiguous.
    Reject questionable order rather than silently changing the experiment.
    """
    sentences = []
    seen_chapters: set[str] = set()
    current_chapter = None
    previous_id = None
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"chapter_id", "sentence_id", "source", "reference"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for line, row in enumerate(reader, start=2):
            nonempty = required - {"reference"} if allow_missing_references else required
            if row.get("reference") is None or any(
                not (row.get(key) or "").strip() for key in nonempty
            ):
                raise ValueError(f"Row {line}: required values cannot be empty")
            chapter = row["chapter_id"].strip()
            try:
                sentence_id = int(row["sentence_id"])
            except ValueError as error:
                raise ValueError(f"Row {line}: sentence_id must be an integer") from error
            if sentence_id < 0:
                raise ValueError(f"Row {line}: sentence_id must be nonnegative")
            if chapter != current_chapter:
                if chapter in seen_chapters:
                    raise ValueError(f"Row {line}: chapter {chapter!r} is not contiguous")
                seen_chapters.add(chapter)
                current_chapter, previous_id = chapter, None
            if previous_id is not None and sentence_id <= previous_id:
                raise ValueError(f"Row {line}: sentence IDs must strictly increase")
            sentences.append(Sentence(chapter, sentence_id, row["source"], row["reference"]))
            previous_id = sentence_id
    if not sentences:
        raise ValueError("Dataset contains no sentences")
    return sentences
