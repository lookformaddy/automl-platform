"""Evaluation metrics and deterministic cross-validation splits.

All metrics are implemented from scratch so the core has no third-party
dependency. ROC-AUC uses the Mann-Whitney rank statistic, which handles
ties correctly; log loss clips probabilities to avoid log(0).
"""

from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

Labels = List[int]


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    if not y_true:
        raise ValueError("empty input")
    return sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)


def precision_recall_f1(
    y_true: Sequence[int], y_pred: Sequence[int]
) -> Tuple[float, float, float]:
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def f1_score(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    return precision_recall_f1(y_true, y_pred)[2]


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Rank-based AUC; returns 0.5 when only one class is present."""
    pairs = sorted(zip(y_score, y_true))
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    # Average ranks for ties (1-based).
    ranks: List[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        i = j + 1
    rank_sum_pos = sum(r for (_, label), r in zip(pairs, ranks) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def log_loss(y_true: Sequence[int], y_proba: Sequence[float], eps: float = 1e-15) -> float:
    total = 0.0
    for label, p in zip(y_true, y_proba):
        p = min(max(p, eps), 1.0 - eps)
        total += -(math.log(p) if label == 1 else math.log(1.0 - p))
    return total / len(y_true)


def stratified_kfold(
    y: Sequence[int], n_splits: int = 5, seed: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Deterministic stratified folds.

    Each fold's test indices are disjoint, every index appears in exactly
    one test fold, and the class mix of each fold mirrors the whole set.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    rng = random.Random(seed)
    by_class: dict[int, List[int]] = {}
    for i, label in enumerate(y):
        by_class.setdefault(label, []).append(i)
    for indices in by_class.values():
        rng.shuffle(indices)
    folds: List[List[int]] = [[] for _ in range(n_splits)]
    for indices in by_class.values():
        for k, idx in enumerate(indices):
            folds[k % n_splits].append(idx)
    result = []
    for k in range(n_splits):
        test = sorted(folds[k])
        test_set = set(test)
        train = [i for i in range(len(y)) if i not in test_set]
        result.append((train, test))
    return result
