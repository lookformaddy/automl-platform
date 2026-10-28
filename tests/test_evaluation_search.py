from automl import SyntheticConfig, generate_classification
from automl.evaluation import accuracy, roc_auc, stratified_kfold
from automl.leaderboard import Leaderboard
from automl.search import SearchConfig, TrialRunner, default_search_space

X, y = generate_classification(SyntheticConfig(n_samples=200, seed=11))


def test_cv_folds_disjoint_deterministic_and_cover_all():
    folds_a = stratified_kfold(y, n_splits=5, seed=3)
    folds_b = stratified_kfold(y, n_splits=5, seed=3)
    assert folds_a == folds_b
    seen = []
    for train, test in folds_a:
        assert not set(train) & set(test)
        seen.extend(test)
    assert sorted(seen) == list(range(len(y)))


def test_metrics_sanity():
    assert accuracy([1, 0, 1], [1, 0, 0]) == 2 / 3
    assert roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert roc_auc([0, 1], [0.5, 0.5]) == 0.5


def test_leaderboard_orders_by_score_then_simplicity():
    runner = TrialRunner(SearchConfig(primary_metric="roc_auc", patience=99))
    trials = runner.run(X, y, default_search_space())
    board = Leaderboard(trials)
    scores = [t.primary_score for t in board.trials]
    assert scores == sorted(scores, reverse=True)
    # Tie-break: equal scores -> simpler model first.
    for a, b in zip(board.trials, board.trials[1:]):
        if a.primary_score == b.primary_score:
            assert a.complexity <= b.complexity
    assert board.champion is not None


def test_early_stopping_triggers_on_stagnant_space():
    # Patience 1 with a space whose later trials cannot improve on the
    # first: repeated identical baselines stagnate immediately.
    runner = TrialRunner(SearchConfig(primary_metric="roc_auc", patience=1, min_delta=0.5))
    trials = runner.run(X, y, [{"model": "majority"}] * 4)
    assert runner.stopped_early is True
    assert len(trials) == 2  # first sets best, second stagnates -> stop
