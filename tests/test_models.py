from automl import SyntheticConfig, generate_classification
from automl.models import (
    DecisionTreeClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    MajorityClassifier,
)

X, y = generate_classification(SyntheticConfig(n_samples=120, seed=7))


def test_each_model_trains_and_predicts():
    for model in [
        MajorityClassifier(),
        DecisionTreeClassifier(max_depth=3),
        DecisionTreeClassifier(max_depth=1),  # stump
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(epochs=100),
    ]:
        model.fit(X, y)
        preds = model.predict(X)
        probas = model.predict_proba(X)
        assert len(preds) == len(y) and len(probas) == len(y)
        assert set(preds) <= {0, 1}
        assert all(0.0 <= p <= 1.0 for p in probas)


def test_logistic_beats_majority_on_signal_data():
    base = MajorityClassifier().fit(X, y)
    lr = LogisticRegression(epochs=200).fit(X, y)
    from automl import accuracy

    assert accuracy(y, lr.predict(X)) > accuracy(y, base.predict(X))


def test_model_params_round_trip():
    for model in [
        MajorityClassifier(),
        DecisionTreeClassifier(max_depth=2),
        KNeighborsClassifier(n_neighbors=3),
        LogisticRegression(epochs=50),
    ]:
        model.fit(X, y)
        restored = type(model).from_params(model.to_params())
        assert restored.predict(X[:10]) == model.predict(X[:10])
