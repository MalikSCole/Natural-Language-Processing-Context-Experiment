"""Assemble checked results and qualitative annotations into a paper draft."""

from collections import Counter
import csv
import json
from pathlib import Path
import re


def main():
    run = Path("results/expanded")
    metadata = json.loads((run / "metadata.json").read_text())
    if metadata["status"] != "complete":
        raise ValueError("Inference is not complete")
    statistics = json.loads((run / "significance.json").read_text())
    plan = statistics["analysis_plan"]
    annotations = json.loads((run / "review_annotations.json").read_text())
    selected = plan["manual_review"]["targets"]
    if set(annotations) != {str(i) for i in range(len(selected))}:
        raise ValueError("Review every predefined target before writing the report")
    with (run / "translations.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with (run / "summary.csv").open(encoding="utf-8", newline="") as handle:
        summary = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        groups.setdefault((row["chapter_id"], int(row["sentence_id"])), {})[int(row["context_size"])] = row
    expected = metadata["target_count"] * 4
    if len(rows) != expected or any(
        not r["prediction"].strip() or r["truncated"] != "False" or
        int(r["actual_context_size"]) != int(r["context_size"]) for r in rows
    ):
        raise ValueError("Output count, nonempty, truncation, or context check failed")
    flags = [{"chapter_id": r["chapter_id"], "sentence_id": r["sentence_id"],
              "context_size": r["context_size"]} for r in rows if re.search(
                  r"(?:PREVIOUS:|CURRENT:|Translation:|<think>|Here is the)", r["prediction"], re.I)]
    review_rows = []
    for i, target in enumerate(selected):
        group = groups[target["chapter_id"], target["sentence_id"]]
        annotation = annotations[str(i)]
        review_rows.append({**target, "source": group[0]["source"],
                            "reference": group[0]["reference"],
                            "prediction_0": group[0]["prediction"],
                            "prediction_3": group[3]["prediction"],
                            "baseline_flags": "; ".join(annotation["baseline_errors"]),
                            "context3_flags": "; ".join(annotation["context3_errors"]),
                            "preference": annotation["preference"], "note": annotation["note"]})
    with (run / "error_review.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)
    preferences = dict(Counter(r["preference"] for r in review_rows))
    preference_text = (f"{preferences.get('context3', 0)} favored context 3, "
                       f"{preferences.get('baseline', 0)} favored baseline, and "
                       f"{preferences.get('tie', 0)} were ties")
    qa = dict(outputs=len(rows), targets=len(groups), empty=0, truncated=0,
              label_scan_flags=flags, manually_reviewed_targets=len(review_rows),
              manually_reviewed_outputs=len(review_rows) * 2, preferences=preferences,
              reviewer="assistant-generated, non-blinded; flags are review concerns, not validated human error rates")
    (run / "quality_checks.json").write_text(json.dumps(qa, indent=2) + "\n")

    table = ["| Previous sentences | BLEU | chrF |", "|---|---:|---:|"]
    for row in summary:
        table.append(f"| {row['context_size']} | {float(row['bleu']):.2f} | {float(row['chrf']):.2f} |")
    tests = ["| Metric | Context 3 − context 0 | 95% bootstrap CI | Two-sided p | Holm-adjusted p |",
             "|---|---:|---|---:|---:|"]
    conclusions = []
    for name, result in statistics["results"].items():
        low, high = result["ci95_difference"]
        tests.append(f"| {name} | {result['difference']:+.2f} | [{low:+.2f}, {high:+.2f}] | {result['p_two_sided']:.4f} | {result['p_holm']:.4f} |")
        if result["significant_at_0_05"] and result["difference"] > 0:
            conclusions.append(f"For {name}, context 3 improved on context 0 under the novel-level randomization test after Holm correction (adjusted p = {result['p_holm']:.4f}).")
        elif result["significant_at_0_05"]:
            conclusions.append(f"For {name}, context 3 scored significantly lower than context 0 after Holm correction (adjusted p = {result['p_holm']:.4f}).")
        else:
            conclusions.append(f"For {name}, the context-3 difference was not statistically significant after Holm correction (adjusted p = {result['p_holm']:.4f}).")
    text = "\n".join([
        "# Results and Discussion — manuscript draft", "", "## Results", "",
        "We evaluated prompt-based context-aware Chinese-to-English translation using a fixed pretrained multilingual instruction model, Qwen3-4B-Instruct-2507, in a fixed 4-bit MLX conversion. No model fine-tuning was performed. The 200 targets were sampled from 20 novels and 40 chapters in the GuoFeng TRAIN split, with five targets per chapter. All targets contained Chinese or ASCII alphanumeric content and had three preceding aligned rows available after the first content row of each chapter was excluded as a presumed heading. Exact inline corpus markers were also removed under rules fixed before generation. Each target was translated under four source-context conditions, yielding 800 translations.", "",
        *table, "",
        "Context 3 achieved the highest BLEU, while the sentence-only baseline achieved the highest chrF. The expanded sample therefore does not show a consistent monotonic improvement as context increases.", "",
        "Scores are corpus-level SacreBLEU BLEU (13a) and chrF (character order 6, word order 0, beta 2), on a 0–100 scale. The model snapshot, prompt and greedy decoding settings were identical across conditions. All 800 outputs were nonempty and untruncated. Exact metric signatures, prompts and run metadata are retained.", "",
        "The primary comparison was context 3 versus context 0, fixed in the analysis plan before this run. We used 10,000 two-sided paired label-swap randomizations at the novel level, retaining all sentences and chapters from each novel as one cluster. The two metric p-values were adjusted with Holm's procedure at alpha .05. Paired novel bootstrap resampling (2,000 replicates) supplied descriptive 95% percentile intervals for corpus-score differences; these intervals are not multiplicity-adjusted.", "",
        *tests, "", *conclusions, "",
        "Contexts 1 and 2 are descriptive comparisons. This analysis was planned before the expanded run but was informed by earlier pilots; it is an exploratory expansion, not an independently preregistered confirmatory study.", "",
        "## Qualitative analysis", "",
        "A fixed subset of 20 targets, one per novel, was reviewed with both baseline and context-3 outputs. The rubric covered names/terminology, pronouns/coreference, agency/negation, omissions/additions, and fluency. The assistant-generated review was not blinded and requires independent human validation. Flags denote review concerns rather than validated error counts. The complete side-by-side annotations are in error_review.csv.", "",
        f"Within this subset, {preference_text}. These counts are not a human evaluation score or an estimate of population-wide improvement. Examples show that context can restore a fragmentary question's framing and a comparison omitted by the baseline. Other examples retain name-spelling ambiguity, omit speech introductions, or introduce less faithful wording. Reference translations can also contain awkward or questionable wording, so lexical overlap alone does not determine translation quality.", "",
        "## Discussion and limitations", "",
        "The comparison isolates the effect of additional source text within a fixed prompt-based translation pipeline. It does not train or evaluate a newly proposed context-aware NMT architecture. Neither primary test supports a statistically significant overall benefit of three preceding rows in this sample. This failure to reject the null is not evidence of equivalence or proof that context never helps individual translations. Novel-level resampling accounts for dependence within the sampled novels, but there are only 20 novel clusters.", "",
        "This is not a held-out benchmark evaluation: the TRAIN split was used for its explicit novel IDs and sufficient coverage. The local held-out files did not provide a verified mapping to the distinct novels needed for this balanced design, and one validation split failed structural checks. Unknown overlap with the instruction model's pretraining material remains possible. The sample is balanced across selected novels rather than representative of sentence frequency or genre. Structural alignment checks and limited manual review cannot rule out reference or alignment errors; excluding problematic chapters also limits coverage.", "",
        "The context unit is an existing aligned corpus row, not a freshly segmented linguistic sentence. The heading filter excludes only each chapter's first content row. Review found a secondary heading retained in one context window; it was not removed after generation. Thus the nominal three-sentence condition can include heading-like text, a preprocessing limitation to address prospectively in future experiments.", "",
        "The earlier 40-target pilot is preserved, including its punctuation-only example. That row was not removed retrospectively. The expanded sample applies the content filter prospectively, and its absolute scores should not be interpreted as a direct before/after comparison with earlier samples. A fuller study should verify held-out novel identities and alignment, add blinded independent human judgments, and assess error categories and reference variation across more novels.", "",
        "## Citation notes", "",
        "BibTeX entries are in reports/references.bib: wang2023guofeng (corpus), yang2025qwen3 and qwen2025instruct2507 (model family and release), post2018clarity (SacreBLEU), and popovic2015chrf (chrF). The exact MLX conversion and immutable revision are recorded in metadata.json. The custom clustered test is described in the analysis plan and README; it should not be described as SacreBLEU's sentence-level CLI test.", ""
    ])
    (run / "report.md").write_text(text)
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    (reports / "acl_results_discussion.md").write_text(text)
    print(f"Report and manuscript draft written; {len(review_rows)} reviewed targets; QA flags: {len(flags)}")


if __name__ == "__main__":
    main()
