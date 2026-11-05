"""Stage 1: prepare identical target sentences for all context conditions."""

import csv
from pathlib import Path

from src.context import CONTEXT_SIZES, build_context
from src.dataset import load_dataset


def prepare_inputs(dataset: Path, output: Path, limit: int = 30) -> int:
    if limit < 1:
        raise ValueError("limit must be positive")
    # Load the ordered corpus before selecting targets to preserve their history.
    sentences = load_dataset(dataset, allow_missing_references=True)
    # Missing references remain in source history but are never scored targets.
    targets = [i for i, row in enumerate(sentences) if row.reference.strip()][:limit]
    if not targets:
        raise ValueError("Dataset contains no targets with references")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["chapter_id", "sentence_id", "source", "reference", "context_size",
              "actual_context_size", "input_text"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in targets:
            for size in CONTEXT_SIZES:
                example = build_context(sentences, index, size)
                sentence = example.sentence
                writer.writerow(dict(
                    chapter_id=sentence.chapter_id, sentence_id=sentence.sentence_id,
                    source=sentence.source, reference=sentence.reference,
                    context_size=size, actual_context_size=len(example.previous_sources),
                    input_text=example.input_text,
                ))
    return len(targets) * len(CONTEXT_SIZES)
