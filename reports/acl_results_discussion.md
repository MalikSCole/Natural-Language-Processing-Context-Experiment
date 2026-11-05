# Results and Discussion — manuscript draft

## Results

We evaluated prompt-based context-aware Chinese-to-English translation using a fixed pretrained multilingual instruction model, Qwen3-4B-Instruct-2507, in a fixed 4-bit MLX conversion. No model fine-tuning was performed. The 200 targets were sampled from 20 novels and 40 chapters in the GuoFeng TRAIN split, with five targets per chapter. All targets contained Chinese or ASCII alphanumeric content and had three preceding aligned rows available after the first content row of each chapter was excluded as a presumed heading. Exact inline corpus markers were also removed under rules fixed before generation. Each target was translated under four source-context conditions, yielding 800 translations.

| Previous sentences | BLEU | chrF |
|---|---:|---:|
| 0 | 13.40 | 40.41 |
| 1 | 13.06 | 40.08 |
| 2 | 13.66 | 39.93 |
| 3 | 14.08 | 40.31 |

Context 3 achieved the highest BLEU, while the sentence-only baseline achieved the highest chrF. The expanded sample therefore does not show a consistent monotonic improvement as context increases.

Scores are corpus-level SacreBLEU BLEU (13a) and chrF (character order 6, word order 0, beta 2), on a 0–100 scale. The model snapshot, prompt and greedy decoding settings were identical across conditions. All 800 outputs were nonempty and untruncated. Exact metric signatures, prompts and run metadata are retained.

The primary comparison was context 3 versus context 0, fixed in the analysis plan before this run. We used 10,000 two-sided paired label-swap randomizations at the novel level, retaining all sentences and chapters from each novel as one cluster. The two metric p-values were adjusted with Holm's procedure at alpha .05. Paired novel bootstrap resampling (2,000 replicates) supplied descriptive 95% percentile intervals for corpus-score differences; these intervals are not multiplicity-adjusted.

| Metric | Context 3 − context 0 | 95% bootstrap CI | Two-sided p | Holm-adjusted p |
|---|---:|---|---:|---:|
| BLEU | +0.68 | [-0.32, +1.99] | 0.2449 | 0.4898 |
| chrF | -0.10 | [-1.24, +0.88] | 0.8528 | 0.8528 |

For BLEU, the context-3 difference was not statistically significant after Holm correction (adjusted p = 0.4898).
For chrF, the context-3 difference was not statistically significant after Holm correction (adjusted p = 0.8528).

Contexts 1 and 2 are descriptive comparisons. This analysis was planned before the expanded run but was informed by earlier pilots; it is an exploratory expansion, not an independently preregistered confirmatory study.

## Qualitative analysis

A fixed subset of 20 targets, one per novel, was reviewed with both baseline and context-3 outputs. The rubric covered names/terminology, pronouns/coreference, agency/negation, omissions/additions, and fluency. The assistant-generated review was not blinded and requires independent human validation. Flags denote review concerns rather than validated error counts. The complete side-by-side annotations are in error_review.csv.

Within this subset, 6 favored context 3, 2 favored baseline, and 12 were ties. These counts are not a human evaluation score or an estimate of population-wide improvement. Examples show that context can restore a fragmentary question's framing and a comparison omitted by the baseline. Other examples retain name-spelling ambiguity, omit speech introductions, or introduce less faithful wording. Reference translations can also contain awkward or questionable wording, so lexical overlap alone does not determine translation quality.

## Discussion and limitations

The comparison isolates the effect of additional source text within a fixed prompt-based translation pipeline. It does not train or evaluate a newly proposed context-aware NMT architecture. Neither primary test supports a statistically significant overall benefit of three preceding rows in this sample. This failure to reject the null is not evidence of equivalence or proof that context never helps individual translations. Novel-level resampling accounts for dependence within the sampled novels, but there are only 20 novel clusters.

This is not a held-out benchmark evaluation: the TRAIN split was used for its explicit novel IDs and sufficient coverage. The local held-out files did not provide a verified mapping to the distinct novels needed for this balanced design, and one validation split failed structural checks. Unknown overlap with the instruction model's pretraining material remains possible. The sample is balanced across selected novels rather than representative of sentence frequency or genre. Structural alignment checks and limited manual review cannot rule out reference or alignment errors; excluding problematic chapters also limits coverage.

The context unit is an existing aligned corpus row, not a freshly segmented linguistic sentence. The heading filter excludes only each chapter's first content row. Review found a secondary heading retained in one context window; it was not removed after generation. Thus the nominal three-sentence condition can include heading-like text, a preprocessing limitation to address prospectively in future experiments.

The earlier 40-target pilot is preserved, including its punctuation-only example. That row was not removed retrospectively. The expanded sample applies the content filter prospectively, and its absolute scores should not be interpreted as a direct before/after comparison with earlier samples. A fuller study should verify held-out novel identities and alignment, add blinded independent human judgments, and assess error categories and reference variation across more novels.

## Citation notes

BibTeX entries are in reports/references.bib: wang2023guofeng (corpus), yang2025qwen3 and qwen2025instruct2507 (model family and release), post2018clarity (SacreBLEU), and popovic2015chrf (chrF). The exact MLX conversion and immutable revision are recorded in metadata.json. The custom clustered test is described in the analysis plan and README; it should not be described as SacreBLEU's sentence-level CLI test.
