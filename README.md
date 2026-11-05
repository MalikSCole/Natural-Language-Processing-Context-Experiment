# Context-aware Chinese-to-English literary translation

Research prototype comparing 0, 1, 2, and 3 preceding Chinese sentences.
Implements dataset validation, chapter-safe context construction, local model
inference, and corpus-level BLEU/chrF scoring.

## Run in VS Code

Open this folder in VS Code. Use Python 3.10 or later; stage 1 has no external
dependencies. From the project root:

```bash
python3 main.py
python3 -m unittest discover -s tests -v
```

The default command prepares 24 synthetic targets across all four conditions
(96 rows) in `results/prepared_inputs.csv`. This is a software fixture with
repeated invented sentences, **not GuoFeng data or research evidence**.

## Folder structure

```text
data/raw/guofeng/     Original corpus files, kept untouched and ignored by Git
data/processed/       Normalized aligned corpus
data/samples/         Small fixtures and future real corpus samples
src/dataset.py        CSV loading and validation
src/context.py        Chapter-safe source context construction
experiments/run_all.py  Prepare all four context conditions
results/translations/  Future model outputs
results/metrics/       Future BLEU and chrF summaries
results/figures/       Future plots
notebooks/            Optional exploration
tests/                Dataset/context verification
main.py               Command-line entry point
```

## Data contract

Obtain the GuoFeng corpus through its authorized distribution and place its
original files in `data/raw/guofeng/`.
The [upstream repository](https://github.com/longyuewangdcu/GuoFeng-Webnovel)
links registration forms for download and documents the Chinese–English V1
format. Its terms require registration and non-commercial research use and
restrict modification/redistribution; review them before obtaining the data.

`src/guofeng.py` imports the documented paired, tagged `.zh`/`.en` format:

```bash
python3 -m src.guofeng --source data/raw/guofeng/train.zh --reference data/raw/guofeng/train.en
```

Filenames above are examples; use the files from your registered download.
The importer retains book/chapter boundaries and pairs existing content lines,
including chapter headings. It rejects structural mismatches, unequal line
counts, and blank source lines rather than guessing alignment. Blank reference
lines retain their source positions for context and are excluded as targets.
Equal line counts cannot prove
semantic alignment; inspect the release and sample pairs before experiments.
Chinese–German/Russian V2 files are not supported by this importer.

### Local corpus validation

The downloaded corpus is in `GuoFeng_Webnovel_Corpus/` and is ignored by Git.
`V1/VALID_1` passes structural validation: 711 paired content rows with no missing
references. The first 30 rows produce 120 inputs across four conditions:

```bash
python3 -m src.guofeng --source GuoFeng_Webnovel_Corpus/V1/VALID_1/valid.zh --reference GuoFeng_Webnovel_Corpus/V1/VALID_1/valid.en --output data/processed/guofeng_valid1.csv
python3 main.py --dataset data/processed/guofeng_valid1.csv --limit 30 --output results/translations/valid1_prepared_inputs.csv
```

`V1/VALID_2` fails chapter-level structure/line-count validation despite equal
total file line counts. It also has three blank reference lines. This split is
excluded pending alignment review; lines are not shifted or dropped to force
a match. Structural validation does not establish semantic alignment quality.

The normalized UTF-8 CSV contract is:

```csv
chapter_id,sentence_id,source,reference
book1:chapter1,0,风停了。,The wind stopped.
book1:chapter1,1,她打开窗户。,She opened the window.
```

Use globally unique chapter IDs, contiguous chapter rows, and strictly
increasing integer sentence IDs within chapters. IDs may start at 0 or 1.
Rows must already be in narrative order. Each row must be an aligned Chinese
sentence and professional English reference; do not independently split the
two languages and assume their sentences align. Empty fields, repeated IDs,
and out-of-order chapters are rejected.

```bash
python3 main.py --dataset data/processed/guofeng.csv --limit 30
```

The first 30 targets are selected deterministically, using the same targets for
all conditions. Chapter beginnings use the available preceding sentences;
`actual_context_size` records this. Context is ordered oldest to newest and
never crosses a chapter boundary. English references are saved for evaluation
and never included in `input_text`.

## Experiment workflow

Normalize the tagged corpus, construct source context, run the fixed pretrained
model, and compute corpus-level metrics. `main.py` only prepares inspectable
inputs. `experiments.translate` performs inference and scoring using the same
context builder, and saves both the exact model prompt and prediction.

Do not commit corpus files without checking redistribution permissions.

## Local model inference and scoring

Use the project Python 3.12 environment on an Apple Silicon Mac:

```bash
source .venv/bin/activate
python -m experiments.download_model
```

The download script resolves and records an immutable revision in
`models/selected.json`. Pass its `path` value to the runner:

```bash
python -m experiments.translate --model /absolute/path/from/models/selected.json --limit 3 --output-dir results/smoke
python -m experiments.translate --model /absolute/path/from/models/selected.json --limit 30 --output-dir results/pilot
python -m unittest discover -s tests -v
```

The chosen model is the 4-bit MLX conversion of Qwen3-4B-Instruct-2507:
https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit
It is an instruction-tuned multilingual language model used for translation,
not a dedicated translation checkpoint. No additional fine-tuning is performed.
The included GuoFeng mBART model is not used for this pilot because its bundled
instructions do not establish how to constrain output to just the current
sentence when given a context window. The Qwen prompt requests that behavior;
manual review is still necessary to assess compliance.

Generation is local, greedy (temperature 0), with a 256-token output limit.
There is no conversation history between calls. All conditions share the same
system prompt and JSON fields; only previous_sentences changes. References
never enter the prompt. Overlong inputs fail instead of being truncated.
Model quantization is fixed across conditions and should be reported in the
paper. Greedy decoding reduces variability but does not guarantee bitwise
reproducibility across hardware or software versions.

Each run writes translations.csv (including the exact prompt), metadata.json,
and summary.csv. Partial runs retain translations for inspection but are not
scored as completed runs. Choose a fresh output directory for each run.
BLEU uses SacreBLEU's 13a tokenizer; chrF uses character order 6, word order 0,
and beta 2. Both are corpus-level scores on a 0–100 scale. Metric signatures
are included. requirements-lock.txt records the installed dependency versions.

These 30 contiguous targets, including a chapter heading, are a software pilot.
They cannot establish that context improves translation in general. Before
larger experiments, review sentence alignment and outputs, select a representative
sample across chapters/books, and predefine how headings are handled.

## Broader exploratory sample

`experiments.prepare_study` scans V1/TRAIN chapter pairs in lockstep, excludes
chapters with unequal line counts, empty source content, or fewer than two
eligible targets, and samples with seed 20261104. The first content row of each
chapter is treated as a heading and removed from body context. Exact inline
`#<#` / `#>#` markers are stripped in both languages before selection and scoring.
These were observed repeatedly in the downloaded data; no other cleanup is
performed. Original files are unchanged.

Sampling is uniform over eligible books (10 selected), then eligible chapters
(two per book via reservoir sampling), then targets (two per chapter). Every
target has three preceding body sentences. The processed CSV retains full
selected chapter bodies so history stays intact. This is book-balanced rather
than a sentence-weighted estimate of the whole corpus. The manifest records
raw and processed hashes, target IDs, exclusions, and the fixed policy.

```bash
python -m experiments.prepare_study
python -m experiments.translate --dataset data/processed/study_train.csv --targets data/processed/study_targets.json --model /absolute/path/from/models/selected.json --output-dir results/study
```

The preparation script refuses to overwrite the frozen manifest. `--targets`
overrides the runner's `--limit`, validates the dataset hash, and uses the exact
same targets for every condition. Sampling does not depend on model outputs.
The local scan excluded 3,327 of 22,567 chapters under these rules.
TRAIN is used here to obtain explicit novel IDs and broader exploratory
coverage; this is not a held-out benchmark claim. Unknown pretraining overlap
is an additional limitation. The original pilot used a different selection and
cleanup policy, so its absolute scores are not directly comparable to this run.

See `results/pilot/error_review.csv` for eight annotated examples of names,
terminology, omissions, and disagreement between reference overlap and quality.

## Expanded 200-target experiment and statistical analysis

The expanded sample is frozen separately; earlier studies are unchanged:

```bash
source .venv/bin/activate
python -m experiments.prepare_study --books 20 --chapters-per-book 2 --targets-per-chapter 5 --seed 20261105 --prefix expanded --require-content
python -m experiments.translate --dataset data/processed/expanded_train.csv --targets data/processed/expanded_targets.json --model /absolute/path/from/models/selected.json --output-dir results/expanded
python -m experiments.analyze --run results/expanded
python -m unittest discover -s tests -v
```

The content filter requires at least one Chinese character (U+3400–U+9FFF),
ASCII letter, or digit in the target. Punctuation-only rows can still provide
original source context but are never translation targets. All 200 targets have
three preceding body sentences. Each novel contributes ten targets, and each
of its two chapters contributes five. Seed, IDs, hashes and policy are saved in
`data/processed/expanded_targets.json`. Do not rerun preparation over this file.

`results/expanded/analysis_plan.json` was written before model generation.
The primary comparison is context 3 versus context 0 on two metrics. Contexts
1 and 2 are descriptive, with no additional significance claims. The primary
two-sided paired randomization test swaps baseline/context-3 output labels for
whole novels (10,000 trials, seed 20261106). This keeps related chapter and
sentence observations together. P-values use the +1 Monte Carlo correction;
Holm correction controls the family of two metric tests at alpha .05.
Descriptive 95% percentile intervals for paired corpus-score differences use
2,000 novel-bootstrap replicates with seed 20261107; these intervals are not
multiplicity-adjusted. This is an exploratory expansion informed by pilots,
not an independently preregistered confirmatory study.

SacreBLEU documents [paired significance testing](https://github.com/mjpost/sacrebleu#paired-significance-tests).
Our test uses the same metric implementations with custom novel-level
resampling, rather than the sentence-level CLI procedure. Cached metric
statistics allow resampling without retokenizing. Their sums are checked against
public corpus_score outputs before testing, and dependency versions are pinned.
Novel-level exchangeability is an assumption; results describe this sample and
sampling scheme, not a proven universal benefit of context.

A held-out file audit is saved in `data/processed/heldout_audit.json`. The local
validation/test wrappers frequently contain a single chapter per BOOK tag, and
some use identical book and chapter ID strings. They do not provide a verified
mapping to the 20 distinct novels needed for this design. VALID_2 also fails
structural alignment. We retain TRAIN with explicit novel IDs, rather than
claiming a broad held-out evaluation. Unknown pretraining overlap remains a
limitation. A verified held-out novel mapping would support a future benchmark.

The manual-review subset is fixed in the analysis plan: first manifest target
per novel, 20 targets total, comparing baseline with context 3. The rubric covers
names/terminology, pronoun/coreference, agency/negation, omissions/additions, and
fluency. This is an assistant-generated, non-blinded qualitative review; it does
not replace blinded independent human ratings.

`requirements-lock.txt` is included at the project root (39 installed packages).
To recreate this Mac/Apple-Silicon Python 3.12 environment:

```bash
uv pip install --python .venv/bin/python -r requirements-lock.txt
```

Use `.venv/bin/python` for tests: running with a different system interpreter
may lack SacreBLEU even when the project environment is correctly installed.
The experiment is prompt-based context-aware translation with a fixed pretrained
instruction model, not training a new context-aware NMT architecture.

After generation and analysis, assemble the annotated review and draft:

```bash
python -m experiments.write_report
```

The expanded run completed all 800 translations without empty or truncated
outputs. Context 3 scored 14.08 BLEU versus 13.40 baseline, and 40.31 chrF versus
40.41 baseline. Neither primary difference was significant after Holm correction
(BLEU p=.4898; chrF p=.8528). The larger sample does not reproduce a consistent
metric improvement as context increases. This does not establish equivalence.

The fixed 20-target qualitative subset contains six context-3 preferences,
two baseline preferences, and twelve ties under assistant review. These are not
blinded human ratings. Review also found a secondary heading retained in a
context window: first-row exclusion is a heuristic, not comprehensive heading
detection. Existing aligned rows, which may contain multiple linguistic
sentences, are the context units. These limitations are disclosed without
retrospectively changing the frozen sample.

`reports/acl_results_discussion.md` is the Results/Discussion draft;
`reports/references.bib` contains bibliography entries. Full statistical outputs,
annotations and run metadata are in `results/expanded/`. The earlier pilots are
preserved. The 16-test suite passes in the project environment, and all 39 lock
entries match its installed package versions.
