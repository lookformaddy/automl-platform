"""Synthetic tabular classification data.

Generates a controllable signal/noise mix: a small number of informative
features carry a linear/logistic signal, the rest are pure noise. Nothing
here is real customer or business data; it exists only so the search,
evaluation, and promotion machinery can be exercised deterministically.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

Features = List[List[float]]
Labels = List[int]


@dataclass(frozen=True)
class SyntheticConfig:
    n_samples: int = 400
    n_features: int = 8
    n_informative: int = 3
    noise_std: float = 1.0
    class_balance: float = 0.5  # target fraction of positives, approx.
    seed: int = 42


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def generate_classification(
    config: SyntheticConfig = SyntheticConfig(),
) -> Tuple[Features, Labels]:
    """Return (X, y) with a deterministic seed.

    Informative features are drawn first, then a logistic score over them
    (with weights that decay so no single feature dominates) decides the
    label; noise features are appended afterwards. The number of positives
    is nudged toward ``class_balance`` by shifting the intercept, which is
    a crude but deterministic calibration for demo purposes.
    """
    if config.n_informative > config.n_features:
        raise ValueError("n_informative must not exceed n_features")
    rng = random.Random(config.seed)
    weights = [1.0 / (i + 1) for i in range(config.n_informative)]
    # Calibrate intercept so positives land near class_balance.
    intercept = math.log(config.class_balance / (1.0 - config.class_balance))
    X: Features = []
    y: Labels = []
    for _ in range(config.n_samples):
        informative = [rng.gauss(0.0, 1.0) for _ in range(config.n_informative)]
        noise = [
            rng.gauss(0.0, config.noise_std)
            for _ in range(config.n_features - config.n_informative)
        ]
        score = intercept + sum(w * v for w, v in zip(weights, informative))
        label = 1 if rng.random() < _sigmoid(score) else 0
        X.append(informative + noise)
        y.append(label)
    return X, y


def train_test_split(
    X: Sequence[Sequence[float]],
    y: Sequence[int],
    test_fraction: float = 0.25,
    seed: int = 42,
) -> Tuple[Features, Features, Labels, Labels]:
    """Deterministic shuffled split (same seed -> same split)."""
    idx = list(range(len(y)))
    random.Random(seed).shuffle(idx)
    n_test = int(round(len(y) * test_fraction))
    test_idx = set(idx[:n_test])
    X_train = [list(X[i]) for i in idx if i not in test_idx]
    y_train = [y[i] for i in idx if i not in test_idx]
    X_test = [list(X[i]) for i in idx if i in test_idx]
    y_test = [y[i] for i in idx if i in test_idx]
    return X_train, X_test, y_train, y_test
