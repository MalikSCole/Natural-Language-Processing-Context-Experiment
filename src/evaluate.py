"""Corpus-level metrics with identical target coverage across conditions."""

import csv
from pathlib import Path


def summarize(rows: list[dict], output: Path) -> list[dict]:
    from sacrebleu.metrics import BLEU, CHRF
    groups = {}
    for row in rows:
        if row["truncated"] in (True, "True", "true", "1"):
            raise ValueError("Truncated outputs must be resolved before scoring")
        if not row["prediction"].strip() or not row["reference"].strip():
            raise ValueError("Empty prediction or reference")
        size = int(row["context_size"])
        key = (row["chapter_id"], str(row["sentence_id"]))
        group = groups.setdefault(size, {})
        if key in group:
            raise ValueError("Duplicate target within a condition")
        group[key] = row
    if set(groups) != {0, 1, 2, 3}:
        raise ValueError("All four conditions are required")
    baseline = groups[0]
    summaries = []
    for size, group in sorted(groups.items()):
        if group.keys() != baseline.keys() or any(
            (group[key]["source"], group[key]["reference"]) !=
            (baseline[key]["source"], baseline[key]["reference"]) for key in baseline
        ):
            raise ValueError("Conditions do not share identical targets and references")
        predictions = [group[key]["prediction"] for key in baseline]
        references = [[baseline[key]["reference"] for key in baseline]]
        bleu, chrf = BLEU(tokenize="13a"), CHRF(char_order=6, word_order=0, beta=2)
        summaries.append(dict(context_size=size, n=len(group),
                              bleu=bleu.corpus_score(predictions, references).score,
                              chrf=chrf.corpus_score(predictions, references).score,
                              bleu_signature=str(bleu.get_signature()),
                              chrf_signature=str(chrf.get_signature())))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return summaries
