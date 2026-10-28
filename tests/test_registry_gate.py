import json

from automl import SyntheticConfig, generate_classification
from automl.leaderboard import Leaderboard
from automl.registry import PromotionGate, Registry, SliceSpec
from automl.search import SearchConfig, TrialRunner, default_search_space

X, y = generate_classification(SyntheticConfig(n_samples=300, seed=21))


def _trials():
    runner = TrialRunner(SearchConfig(primary_metric="roc_auc", patience=99))
    trials = runner.run(X, y, default_search_space())
    baseline = next(t for t in trials if t.config["model"] == "majority")
    return trials, baseline


def test_gate_passes_strong_champion(tmp_path):
    trials, baseline = _trials()
    champion = Leaderboard(trials).champion
    gate = PromotionGate(min_margin=0.05, slices=[SliceSpec(0, 0.0, "f0<=0")])
    result = gate.check(champion, baseline, X, y)
    assert result.passed, result.reasons


def test_gate_blocks_weak_model():
    trials, baseline = _trials()
    weak = baseline  # a model cannot beat itself by the margin
    gate = PromotionGate(min_margin=0.05)
    result = gate.check(weak, baseline, X, y)
    assert not result.passed
    assert any("margin" in r for r in result.reasons)


def test_gate_blocks_slice_collapse(tmp_path):
    trials, baseline = _trials()
    champion = Leaderboard(trials).champion
    # Zero tolerance on a slice where the champion is at most as good as
    # baseline is hard to force; instead use an impossible margin-free
    # check via a huge tolerance inversion: slice check with tolerance
    # so tight it can only fail if candidate < baseline on that slice.
    # Simpler deterministic check: baseline vs itself with a negative
    # tolerance must fail on any non-empty slice.
    gate = PromotionGate(min_margin=0.0, slice_tolerance=-0.0001,
                         slices=[SliceSpec(0, 0.0, "f0<=0")])
    result = gate.check(baseline, baseline, X, y)
    assert not result.passed


def test_registry_round_trip(tmp_path):
    trials, _ = _trials()
    champion = Leaderboard(trials).champion
    path = tmp_path / "registry.json"
    registry = Registry(path)
    registry.register(champion)
    assert champion.name in registry.list_champions()
    reloaded = Registry(path).load_model(champion.name)
    assert reloaded.predict(X[:20]) == champion.model.predict(X[:20])
    assert json.loads(path.read_text())[champion.name]["metrics"]
