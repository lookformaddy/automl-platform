"""End-to-end demo: synthetic data -> search -> leaderboard -> gate -> registry.

Run:  python3 demo.py
The registry is written to a temporary directory so the demo leaves no
artifacts in the repo.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from automl import (  # noqa: E402
    Leaderboard,
    PromotionGate,
    Registry,
    SearchConfig,
    SyntheticConfig,
    TrialRunner,
    default_search_space,
    generate_classification,
)
from automl.datasets import train_test_split  # noqa: E402
from automl.registry import SliceSpec  # noqa: E402


def main() -> None:
    X, y = generate_classification(SyntheticConfig(n_samples=400, seed=42))
    X_train, X_test, y_train, y_test = train_test_split(X, y, seed=42)
    runner = TrialRunner(SearchConfig(primary_metric="roc_auc", patience=3))
    trials = runner.run(X_train, y_train, default_search_space())
    board = Leaderboard(trials)
    print(f"Trials run: {runner.trials_run} (stopped early: {runner.stopped_early})")
    print(f"{'rank':<5}{'trial':<24}{'roc_auc':<10}{'accuracy':<10}{'f1':<8}")
    for row in board.rows():
        print(
            f"{row['rank']:<5}{row['trial']:<24}"
            f"{row['metrics']['roc_auc']:<10.4f}"
            f"{row['metrics']['accuracy']:<10.4f}"
            f"{row['metrics']['f1']:<8.4f}"
        )
    champion = board.champion
    baseline = next(t for t in trials if t.config["model"] == "majority")
    assert champion is not None
    gate = PromotionGate(
        min_margin=0.05,
        slices=[SliceSpec(feature=0, threshold=0.0, name="feature0<=0")],
    )
    result = gate.check(champion, baseline, X_test, y_test)
    print(f"\nChampion: {champion.name} (roc_auc={champion.primary_score:.4f})")
    print(f"Promotion gate passed: {result.passed}")
    for reason in result.reasons:
        print(f"  - {reason}")
    if result.passed:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Registry(os.path.join(tmp, "registry.json"))
            registry.register(champion)
            reloaded = registry.load_model(champion.name)
            print(
                "Registry round-trip OK:",
                reloaded.predict(X_test[:3]) == champion.model.predict(X_test[:3]),
            )


if __name__ == "__main__":
    main()
