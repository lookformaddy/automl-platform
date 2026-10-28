"""Search space, trial runner, and early stopping.

A trial = one candidate config evaluated by k-fold CV on a primary
metric. Trials run in declaration order; early stopping halts the search
when the best score has not improved by more than ``min_delta`` for
``patience`` consecutive trials. That mirrors a real budget guard: stop
spending compute on a stagnant search space instead of exhausting it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from automl.evaluation import (
    accuracy,
    f1_score,
    log_loss,
    roc_auc,
    stratified_kfold,
)
from automl.models import BaseModel, build_model


def default_search_space() -> List[Dict[str, Any]]:
    """Small, diverse, deterministic space: baseline first, then learners."""
    return [
        {"model": "majority"},
        {"model": "tree", "max_depth": 1, "min_samples_split": 10},
        {"model": "tree", "max_depth": 3, "min_samples_split": 10},
        {"model": "tree", "max_depth": 5, "min_samples_split": 20},
        {"model": "logistic", "learning_rate": 0.1, "epochs": 300, "l2": 0.01},
        {"model": "logistic", "learning_rate": 0.3, "epochs": 300, "l2": 0.1},
        {"model": "knn", "n_neighbors": 3},
        {"model": "knn", "n_neighbors": 9},
    ]


METRIC_FNS = {
    "accuracy": lambda y, p, s: accuracy(y, p),
    "f1": lambda y, p, s: f1_score(y, p),
    "roc_auc": lambda y, p, s: roc_auc(y, s),
    "neg_log_loss": lambda y, p, s: -log_loss(y, s),
}


@dataclass
class TrialResult:
    name: str
    config: Dict[str, Any]
    primary_score: float
    metrics: Dict[str, float]
    model: BaseModel = field(repr=False)

    @property
    def complexity(self) -> int:
        return self.model.complexity


@dataclass
class SearchConfig:
    primary_metric: str = "roc_auc"
    n_splits: int = 5
    seed: int = 42
    patience: int = 3  # trials without improvement before stopping
    min_delta: float = 0.005


class TrialRunner:
    def __init__(self, config: SearchConfig = SearchConfig()) -> None:
        if config.primary_metric not in METRIC_FNS:
            raise ValueError(f"unknown metric: {config.primary_metric}")
        self.config = config
        self.stopped_early: bool = False
        self.trials_run: int = 0

    def run(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        search_space: List[Dict[str, Any]],
    ) -> List[TrialResult]:
        folds = stratified_kfold(y, n_splits=self.config.n_splits, seed=self.config.seed)
        results: List[TrialResult] = []
        best = float("-inf")
        stagnant = 0
        for i, cfg in enumerate(search_space):
            metric_sums: Dict[str, float] = {
                "accuracy": 0.0,
                "f1": 0.0,
                "roc_auc": 0.0,
                "neg_log_loss": 0.0,
            }
            final_model: BaseModel | None = None
            for train_idx, test_idx in folds:
                X_tr = [list(X[j]) for j in train_idx]
                y_tr = [y[j] for j in train_idx]
                X_te = [list(X[j]) for j in test_idx]
                y_te = [y[j] for j in test_idx]
                model = build_model(cfg)
                model.fit(X_tr, y_tr)
                final_model = model
                preds = model.predict(X_te)
                probas = model.predict_proba(X_te)
                metric_sums["accuracy"] += accuracy(y_te, preds)
                metric_sums["f1"] += f1_score(y_te, preds)
                metric_sums["roc_auc"] += roc_auc(y_te, probas)
                metric_sums["neg_log_loss"] += -log_loss(y_te, probas)
            # Refit on the full data so the trial's model is usable.
            final_model = build_model(cfg)
            final_model.fit([list(r) for r in X], list(y))
            metrics = {k: v / len(folds) for k, v in metric_sums.items()}
            primary = metrics[self.config.primary_metric]
            results.append(
                TrialResult(
                    name=f"trial-{i}-{cfg['model']}",
                    config=cfg,
                    primary_score=primary,
                    metrics=metrics,
                    model=final_model,
                )
            )
            self.trials_run += 1
            if primary > best + self.config.min_delta:
                best = primary
                stagnant = 0
            else:
                stagnant += 1
                if stagnant >= self.config.patience:
                    self.stopped_early = True
                    break
        return results
