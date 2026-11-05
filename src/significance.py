"""Paired novel-cluster inference using cached SacreBLEU corpus statistics.

Whole novels are resampled/swapped, preserving within-novel dependence. This
is a custom clustered test, not SacreBLEU's sentence-level CLI test. Pin the
SacreBLEU version: its private cached-statistics interface is checked against
the public corpus_score result before any resampling.
"""

import numpy as np


def holm_adjust(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def paired_cluster_test(metric, baseline, treatment, references, clusters,
                        permutations=10000, bootstrap=2000,
                        permutation_seed=20261106, bootstrap_seed=20261107):
    if not baseline or not (len(baseline) == len(treatment) == len(references) == len(clusters)):
        raise ValueError("Aligned, nonempty paired arrays are required")
    if permutations < 1 or bootstrap < 2:
        raise ValueError("Require positive permutations and at least two bootstrap replicates")
    labels = sorted(set(clusters))
    base_stats = np.array(metric._extract_corpus_statistics(baseline, [references]), dtype=np.int64)
    test_stats = np.array(metric._extract_corpus_statistics(treatment, [references]), dtype=np.int64)

    def score(stats):
        return metric._compute_score_from_stats(stats.tolist()).score

    for predictions, stats in [(baseline, base_stats), (treatment, test_stats)]:
        if not np.isclose(score(stats.sum(axis=0)), metric.corpus_score(predictions, [references]).score,
                          rtol=0, atol=1e-10):
            raise ValueError("Cached metric statistics disagree with public corpus_score")
    base = np.stack([base_stats[np.array(clusters) == label].sum(axis=0) for label in labels])
    test = np.stack([test_stats[np.array(clusters) == label].sum(axis=0) for label in labels])
    base_total, test_total = base.sum(axis=0), test.sum(axis=0)
    baseline_score, treatment_score = score(base_total), score(test_total)
    observed = treatment_score - baseline_score
    rng = np.random.default_rng(permutation_seed)
    extreme = 0
    for _ in range(permutations):
        swaps = rng.integers(0, 2, len(labels))[:, None]
        change = ((test - base) * swaps).sum(axis=0)
        difference = score(test_total - change) - score(base_total + change)
        extreme += abs(difference) >= abs(observed) - 1e-12
    rng = np.random.default_rng(bootstrap_seed)
    differences = []
    for _ in range(bootstrap):
        indices = rng.integers(0, len(labels), len(labels))
        differences.append(score(test[indices].sum(axis=0)) - score(base[indices].sum(axis=0)))
    low, high = np.quantile(differences, [0.025, 0.975])
    return dict(baseline=baseline_score, context_3=treatment_score, difference=observed,
                p_two_sided=float((extreme + 1) / (permutations + 1)),
                ci95_difference=[float(low), float(high)], clusters=len(labels), targets=len(baseline),
                signature=str(metric.get_signature()))
