"""Candidate models, implemented from scratch (no sklearn).

Every model shares the same interface: ``fit(X, y)``, ``predict(X)``,
``predict_proba(X)`` (probability of class 1), ``to_params()`` and
``from_params()`` so the registry can serialize a champion to JSON.
Complexity ranks feed the leaderboard tie-break: simpler wins ties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

Features = List[List[float]]
Labels = List[int]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class BaseModel:
    model_type: str = "base"
    complexity: int = 99

    def fit(self, X: Features, y: Labels) -> "BaseModel":
        raise NotImplementedError

    def predict(self, X: Features) -> Labels:
        return [1 if p >= 0.5 else 0 for p in self.predict_proba(X)]

    def predict_proba(self, X: Features) -> List[float]:
        raise NotImplementedError

    def to_params(self) -> Dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "BaseModel":
        raise NotImplementedError


class MajorityClassifier(BaseModel):
    """Baseline: always predicts the majority training class."""

    model_type = "majority"
    complexity = 0

    def __init__(self) -> None:
        self.majority_class: int = 0

    def fit(self, X: Features, y: Labels) -> "MajorityClassifier":
        self.majority_class = 1 if sum(y) * 2 >= len(y) else 0
        return self

    def predict_proba(self, X: Features) -> List[float]:
        p = 1.0 if self.majority_class == 1 else 0.0
        return [p for _ in X]

    def to_params(self) -> Dict[str, Any]:
        return {"model_type": self.model_type, "majority_class": self.majority_class}

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "MajorityClassifier":
        m = cls()
        m.majority_class = int(params["majority_class"])
        return m


class KNeighborsClassifier(BaseModel):
    """k-NN with Euclidean distance; proba = neighbour positive fraction."""

    model_type = "knn"
    complexity = 2

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors
        self._X: Features = []
        self._y: Labels = []

    def fit(self, X: Features, y: Labels) -> "KNeighborsClassifier":
        self._X = [list(row) for row in X]
        self._y = list(y)
        return self

    def predict_proba(self, X: Features) -> List[float]:
        out = []
        for row in X:
            dists = sorted(
                (sum((a - b) ** 2 for a, b in zip(row, train)), label)
                for train, label in zip(self._X, self._y)
            )
            k = min(self.n_neighbors, len(dists))
            out.append(sum(label for _, label in dists[:k]) / k if k else 0.0)
        return out

    def to_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "n_neighbors": self.n_neighbors,
            "X": self._X,
            "y": self._y,
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "KNeighborsClassifier":
        m = cls(n_neighbors=int(params["n_neighbors"]))
        m._X = [list(r) for r in params["X"]]
        m._y = [int(v) for v in params["y"]]
        return m


@dataclass
class _TreeNode:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional["_TreeNode"] = None
    right: Optional["_TreeNode"] = None
    proba: Optional[float] = None  # set on leaves only

    def to_dict(self) -> Dict[str, Any]:
        if self.proba is not None:
            return {"proba": self.proba}
        assert self.left is not None and self.right is not None
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "_TreeNode":
        if "proba" in d:
            return cls(proba=float(d["proba"]))
        return cls(
            feature=int(d["feature"]),  # type: ignore[arg-type]
            threshold=float(d["threshold"]),  # type: ignore[arg-type]
            left=cls.from_dict(d["left"]),
            right=cls.from_dict(d["right"]),
        )


def _gini(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    p = sum(labels) / len(labels)
    return 2 * p * (1 - p)


class DecisionTreeClassifier(BaseModel):
    """Greedy CART tree on Gini impurity, depth- and size-limited."""

    model_type = "tree"
    complexity = 3

    def __init__(self, max_depth: int = 3, min_samples_split: int = 10) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self._root: Optional[_TreeNode] = None

    def fit(self, X: Features, y: Labels) -> "DecisionTreeClassifier":
        self._root = self._build(X, y, depth=0)
        return self

    def _build(self, X: Features, y: Labels, depth: int) -> _TreeNode:
        proba = sum(y) / len(y) if y else 0.0
        if (
            depth >= self.max_depth
            or len(y) < self.min_samples_split
            or len(set(y)) == 1
        ):
            return _TreeNode(proba=proba)
        best: Optional[Tuple[float, int, float]] = None  # (gain, feat, thr)
        n = len(y)
        base = _gini(y)
        for feat in range(len(X[0])):
            values = sorted(set(row[feat] for row in X))
            for a, b in zip(values, values[1:]):
                thr = (a + b) / 2.0
                left = [label for row, label in zip(X, y) if row[feat] <= thr]
                right = [label for row, label in zip(X, y) if row[feat] > thr]
                if not left or not right:
                    continue
                gain = base - (len(left) * _gini(left) + len(right) * _gini(right)) / n
                if best is None or gain > best[0]:
                    best = (gain, feat, thr)
        if best is None or best[0] <= 0:
            return _TreeNode(proba=proba)
        _, feat, thr = best
        left_X = [r for r in X if r[feat] <= thr]
        left_y = [label for r, label in zip(X, y) if r[feat] <= thr]
        right_X = [r for r in X if r[feat] > thr]
        right_y = [label for r, label in zip(X, y) if r[feat] > thr]
        return _TreeNode(
            feature=feat,
            threshold=thr,
            left=self._build(left_X, left_y, depth + 1),
            right=self._build(right_X, right_y, depth + 1),
        )

    def _proba_one(self, row: Sequence[float]) -> float:
        node = self._root
        assert node is not None
        while node.proba is None:
            assert node.left is not None and node.right is not None
            node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[operator]
        return node.proba

    def predict_proba(self, X: Features) -> List[float]:
        return [self._proba_one(row) for row in X]

    def to_params(self) -> Dict[str, Any]:
        assert self._root is not None
        return {
            "model_type": self.model_type,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "tree": self._root.to_dict(),
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "DecisionTreeClassifier":
        m = cls(
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
        )
        m._root = _TreeNode.from_dict(params["tree"])
        return m


class LogisticRegression(BaseModel):
    """Full-batch gradient descent on standardized features, with L2."""

    model_type = "logistic"
    complexity = 4

    def __init__(
        self, learning_rate: float = 0.1, epochs: int = 300, l2: float = 0.01
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.mean: List[float] = []
        self.std: List[float] = []
        self.weights: List[float] = []
        self.bias: float = 0.0

    def _scale(self, X: Features) -> Features:
        return [
            [(v - m) / s for v, m, s in zip(row, self.mean, self.std)] for row in X
        ]

    def fit(self, X: Features, y: Labels) -> "LogisticRegression":
        n_feat = len(X[0])
        self.mean = [sum(row[j] for row in X) / len(X) for j in range(n_feat)]
        self.std = []
        for j in range(n_feat):
            var = sum((row[j] - self.mean[j]) ** 2 for row in X) / len(X)
            self.std.append(math.sqrt(var) if var > 0 else 1.0)
        Xs = self._scale(X)
        self.weights = [0.0] * n_feat
        self.bias = 0.0
        n = len(Xs)
        for _ in range(self.epochs):
            grad_w = [0.0] * n_feat
            grad_b = 0.0
            for row, label in zip(Xs, y):
                p = _sigmoid(sum(w * v for w, v in zip(self.weights, row)) + self.bias)
                err = p - label
                for j in range(n_feat):
                    grad_w[j] += err * row[j]
                grad_b += err
            for j in range(n_feat):
                self.weights[j] -= self.learning_rate * (
                    grad_w[j] / n + self.l2 * self.weights[j]
                )
            self.bias -= self.learning_rate * grad_b / n
        return self

    def predict_proba(self, X: Features) -> List[float]:
        Xs = self._scale(X)
        return [
            _sigmoid(sum(w * v for w, v in zip(self.weights, row)) + self.bias)
            for row in Xs
        ]

    def to_params(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "l2": self.l2,
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "LogisticRegression":
        m = cls(
            learning_rate=float(params["learning_rate"]),
            epochs=int(params["epochs"]),
            l2=float(params["l2"]),
        )
        m.mean = [float(v) for v in params["mean"]]
        m.std = [float(v) for v in params["std"]]
        m.weights = [float(v) for v in params["weights"]]
        m.bias = float(params["bias"])
        return m


MODEL_CLASSES: Dict[str, Any] = {
    cls.model_type: cls
    for cls in (
        MajorityClassifier,
        KNeighborsClassifier,
        DecisionTreeClassifier,
        LogisticRegression,
    )
}


def build_model(config: Dict[str, Any]) -> BaseModel:
    """Build a model from a search-space config dict.

    ``{"model": "tree", "max_depth": 3, ...}`` — the ``model`` key selects
    the class; remaining keys are constructor kwargs.
    """
    cfg = dict(config)
    model_type = cfg.pop("model")
    cls = MODEL_CLASSES[model_type]
    return cls(**cfg)  # type: ignore[operator]
