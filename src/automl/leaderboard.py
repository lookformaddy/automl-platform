"""Leaderboard: rank trials by the primary metric, with tie-breaks.

Ordering: higher primary score first; on an exact tie, the simpler model
wins; remaining ties break alphabetically by trial name so the ranking
is fully deterministic and reproducible across runs.
"""

from __future__ import annotations

from typing import List

from automl.search import TrialResult


class Leaderboard:
    def __init__(self, trials: List[TrialResult]) -> None:
        self.trials = sorted(
            trials,
            key=lambda t: (-t.primary_score, t.complexity, t.name),
        )

    @property
    def champion(self) -> TrialResult | None:
        return self.trials[0] if self.trials else None

    def top(self, n: int = 5) -> List[TrialResult]:
        return self.trials[:n]

    def rows(self) -> List[dict]:
        return [
            {
                "rank": i + 1,
                "trial": t.name,
                "config": t.config,
                "primary_score": round(t.primary_score, 4),
                "metrics": {k: round(v, 4) for k, v in t.metrics.items()},
            }
            for i, t in enumerate(self.trials)
        ]
