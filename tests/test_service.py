import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # TestClient transport

from fastapi.testclient import TestClient  # noqa: E402

from automl.models import LogisticRegression  # noqa: E402
from automl.service import create_app, set_champion  # noqa: E402
from automl import SyntheticConfig, generate_classification  # noqa: E402


def test_predict_endpoint():
    X, y = generate_classification(SyntheticConfig(n_samples=100, seed=5))
    model = LogisticRegression(epochs=50).fit(X, y)
    set_champion(model, name="test-champion")
    client = TestClient(create_app())
    assert client.get("/health").json()["status"] == "ok"
    resp = client.post("/predict", json={"features": X[:3]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 3
    assert len(body["probabilities"]) == 3
