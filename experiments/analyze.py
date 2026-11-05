"""Execute the frozen primary analysis for the expanded experiment."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sacrebleu.metrics import BLEU, CHRF

from src.evaluate import summarize
from src.significance import paired_cluster_test, holm_adjust


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/expanded"))
    args = parser.parse_args()
    metadata = json.loads((args.run / "metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("Inference must finish before analysis")
    plan = json.loads((args.run / "analysis_plan.json").read_text())
    manifest = metadata["target_manifest"]
    original_manifest = Path("data/processed/expanded_targets.json")
    if hashlib.sha256(original_manifest.read_bytes()).hexdigest() != plan["target_manifest_sha256"]:
        raise ValueError("Frozen manifest hash differs from analysis plan")
    if json.loads(original_manifest.read_text()) != manifest:
        raise ValueError("Run used a different manifest")
    with (args.run / "translations.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summarize(rows, args.run / "summary.csv")  # Validate paired coverage and truncation.
    groups = {size: {} for size in range(4)}
    for row in rows:
        groups[int(row["context_size"])][row["chapter_id"], int(row["sentence_id"])] = row
    keys = [(t["chapter_id"], t["sentence_id"]) for t in manifest["targets"]]
    if any(set(group) != set(keys) for group in groups.values()):
        raise ValueError("Generated targets differ from the frozen manifest")
    baseline = [groups[0][key]["prediction"] for key in keys]
    treatment = [groups[3][key]["prediction"] for key in keys]
    references = [groups[0][key]["reference"] for key in keys]
    clusters = [t["book_id"] for t in manifest["targets"]]
    results = {}
    for name, metric in [("BLEU", BLEU(tokenize="13a")),
                         ("chrF", CHRF(char_order=6, word_order=0, beta=2))]:
        results[name] = paired_cluster_test(
            metric, baseline, treatment, references, clusters,
            permutations=plan["randomization"]["trials"],
            bootstrap=plan["bootstrap"]["replicates"],
            permutation_seed=plan["randomization"]["seed"],
            bootstrap_seed=plan["bootstrap"]["seed"])
    adjusted = holm_adjust([r["p_two_sided"] for r in results.values()])
    for result, p in zip(results.values(), adjusted):
        result["p_holm"] = p
        result["significant_at_0_05"] = p < plan["randomization"]["alpha"]
    output = dict(method="paired novel-cluster label-swap randomization; percentile paired novel bootstrap",
                  analysis_plan=plan, results=results)
    (args.run / "significance.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
