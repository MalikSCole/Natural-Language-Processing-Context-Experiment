"""Run one fixed local model across all four context conditions."""

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

from src.context import build_context, CONTEXT_SIZES
from src.dataset import load_dataset
from src.evaluate import summarize
from src.translator import Translator, MODEL_ID, SYSTEM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/guofeng_valid1.csv"))
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--targets", type=Path, help="Frozen target manifest; overrides --limit")
    parser.add_argument("--model", required=True, help="Local pinned Hugging Face snapshot path")
    parser.add_argument("--output-dir", type=Path, default=Path("results/pilot"))
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "translations.csv"
    if predictions_path.exists():
        parser.error("Output already exists; choose a new --output-dir")
    sentences = load_dataset(args.dataset, allow_missing_references=True)
    targets = [i for i, row in enumerate(sentences) if row.reference.strip()][:args.limit]
    if args.targets:
        manifest = json.loads(args.targets.read_text())
        if manifest["dataset_sha256"] != hashlib.sha256(args.dataset.read_bytes()).hexdigest():
            parser.error("Manifest does not match dataset hash")
        lookup = {(r.chapter_id, r.sentence_id): i for i, r in enumerate(sentences)}
        keys = [(r["chapter_id"], r["sentence_id"]) for r in manifest["targets"]]
        if len(set(keys)) != len(keys) or any(key not in lookup for key in keys):
            parser.error("Manifest contains duplicate or unknown targets")
        targets = [lookup[key] for key in keys]
        if any(not sentences[i].reference.strip() for i in targets):
            parser.error("Target has no reference")
    if not targets:
        parser.error("No eligible targets")
    metadata = dict(model=MODEL_ID, snapshot=str(Path(args.model).resolve()),
                    quantization="4bit", temperature=0.0, max_new_tokens=256,
                    max_total_tokens=4096, truncation_policy="fail", system_prompt=SYSTEM,
                    dataset=str(args.dataset), dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
                    target_count=len(targets), python=platform.python_version(),
                    packages={p: version(p) for p in ["mlx-lm", "mlx", "transformers", "sacrebleu"]},
                    status="running")
    if args.targets:
        metadata["target_manifest"] = manifest
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    translator = Translator(args.model)
    rows = []
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["chapter_id", "sentence_id", "source", "reference", "context_size",
                  "actual_context_size", "prediction", "prompt", "input_tokens", "output_tokens", "truncated"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in targets:
            for size in CONTEXT_SIZES:
                example = build_context(sentences, index, size)
                row = dict(chapter_id=example.sentence.chapter_id,
                           sentence_id=example.sentence.sentence_id,
                           source=example.sentence.source, reference=example.sentence.reference,
                           context_size=size, actual_context_size=len(example.previous_sources),
                           **translator.translate(example))
                rows.append(row)
                writer.writerow(row)
                handle.flush()
                print(f"Translated {len(rows)}/{len(targets) * 4}", flush=True)
    scores = summarize(rows, args.output_dir / "summary.csv")
    for size in CONTEXT_SIZES:
        with (args.output_dir / f"context_{size}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(row for row in rows if row["context_size"] == size)
    metadata["status"] = "complete"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    for row in scores:
        print(f"Context {row['context_size']}: BLEU {row['bleu']:.2f}, chrF {row['chrf']:.2f}")


if __name__ == "__main__":
    main()
