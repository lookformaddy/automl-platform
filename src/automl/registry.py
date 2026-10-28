"""Champion promotion gate and model registry.

A trial becomes champion only if it clears BOTH gates:
1. Margin gate — it must beat the baseline trial's primary score by at
   least ``min_margin`` on the same evaluation.
2. Slice gate — on every declared slice (a feature-threshold split of the
   evaluation set), its accuracy must not fall more than ``slice_tolerance``
   below the baseline's accuracy on that slice. A model that wins on
   average but collapses on a subgroup does not ship.

The registry stores champion artifacts as JSON (model params + metrics),
so a champion can be reloaded without retraining.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from automl.evaluation import accuracy
from automl.models import MODEL_CLASSES, BaseModel
from automl.search import TrialResult


@dataclass(frozen=True)
class SliceSpec:
    feature: int
    threshold: float
    name: str = ""


@dataclass
class GateResult:
    passed: bool
    reasons: List[str]


class PromotionGate:
    def __init__(
        self,
        min_margin: float = 0.05,
        slice_tolerance: float = 0.10,
        slices: Sequence[SliceSpec] = (),
    ) -> None:
        self.min_margin = min_margin
        self.slice_tolerance = slice_tolerance
        self.slices = list(slices)

    def check(
        self,
        candidate: TrialResult,
        baseline: TrialResult,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
    ) -> GateResult:
        reasons: List[str] = []
        margin = candidate.primary_score - baseline.primary_score
        if margin < self.min_margin:
            reasons.append(
                f"margin {margin:.4f} < required {self.min_margin:.4f} "
                f"over baseline {baseline.name}"
            )
        cand_preds = candidate.model.predict([list(r) for r in X])
        base_preds = baseline.model.predict([list(r) for r in X])
        for spec in self.slices:
            idx = [i for i, row in enumerate(X) if row[spec.feature] <= spec.threshold]
            if not idx:
                continue
            cand_acc = accuracy([y[i] for i in idx], [cand_preds[i] for i in idx])
            base_acc = accuracy([y[i] for i in idx], [base_preds[i] for i in idx])
            if cand_acc < base_acc - self.slice_tolerance:
                reasons.append(
                    f"slice '{spec.name or spec.feature}' accuracy "
                    f"{cand_acc:.4f} vs baseline {base_acc:.4f} "
                    f"(tolerance {self.slice_tolerance:.4f})"
                )
        return GateResult(passed=not reasons, reasons=reasons)


@dataclass
class ChampionRecord:
    name: str
    model_params: Dict[str, Any]
    metrics: Dict[str, float]
    config: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_params": self.model_params,
            "metrics": self.metrics,
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ChampionRecord":
        return cls(
            name=d["name"],
            model_params=d["model_params"],
            metrics=d["metrics"],
            config=d["config"],
        )


class Registry:
    """JSON-file registry holding at most one champion per name."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: Dict[str, ChampionRecord] = {}
        if self.path.exists():
            data = json.loads(self.path.read_text())
            for name, rec in data.items():
                self._records[name] = ChampionRecord.from_dict(rec)

    def register(self, trial: TrialResult) -> ChampionRecord:
        record = ChampionRecord(
            name=trial.name,
            model_params=trial.model.to_params(),
            metrics=trial.metrics,
            config=trial.config,
        )
        self._records[trial.name] = record
        self._save()
        return record

    def load_model(self, name: str) -> BaseModel:
        record = self._records[name]
        model_type = record.model_params["model_type"]
        return MODEL_CLASSES[model_type].from_params(record.model_params)

    def list_champions(self) -> List[str]:
        return sorted(self._records)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: v.to_dict() for k, v in self._records.items()}, indent=2)
        )
