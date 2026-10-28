# automl-platform

A tabular AutoML platform built from scratch — no sklearn, no black-box
AutoML library. It answers the enterprise question: **how do you pick a
model you can defend shipping?** By running a deterministic candidate
search, ranking it on a leaderboard, and promoting a champion only when
it clears explicit gates.

## The enterprise problem

Teams choosing a model by "best validation score" ship models that win on
average and fail on a subgroup, or beat a baseline by a rounding error
and add operational complexity for nothing. This platform makes the
promotion decision explicit and auditable:

1. **Candidate search** — logistic regression, decision trees/stumps,
   k-NN, and a majority baseline, each from scratch, evaluated by
   stratified k-fold cross-validation on a primary metric.
2. **Early stopping** — the search halts when the best score stagnates
   for `patience` trials, instead of burning budget on a dead space.
3. **Leaderboard** — ranked by the primary metric, ties broken toward
   the simpler model, then alphabetically, so rankings are reproducible.
4. **Promotion gate** — a champion must beat the baseline by a margin
   *and* pass per-slice accuracy checks (no subgroup collapse).
5. **Registry** — champions are stored as JSON-serializable parameters
   and reloadable without retraining.
6. **Prediction service** — a small FastAPI service serving the champion.

All data is synthetic (see `src/automl/datasets.py`): informative
features carry a controllable signal, the rest are noise. No real
customer, business, or employer data is involved.

## Architecture overview

```
synthetic data ──> TrialRunner (stratified k-fold CV, early stopping)
                        │  candidates: majority / tree / k-NN / logistic
                        ▼
                   Leaderboard (primary metric, simplicity tie-break)
                        │  champion candidate
                        ▼
                   PromotionGate (margin gate + slice gates)
                        │  passes
                        ▼
                   Registry (JSON champion artifacts)
                        │
                        ▼
                   Prediction service (FastAPI, optional)
```

See `docs/architecture.md` for the full description and diagram.

## How to run

```bash
python3 -m pip install -r requirements.txt   # pytest (+ fastapi/uvicorn for the service)
python3 -m pytest                            # core tests; service test skips if FastAPI is absent
python3 demo.py                              # end-to-end search + leaderboard + gate + registry
uvicorn app:app --reload                     # prediction service (needs fastapi)
```

The core library has **no third-party dependencies** — pure standard
library Python — so `pytest` passes on plain system Python.

## Key design decisions and trade-offs

- **From-scratch models over sklearn.** The point of the portfolio piece
  is demonstrating the internals: gradient descent, Gini splits,
  rank-based AUC. The cost is scale — this is for tabular problems in
  the thousands of rows, not millions. A production deployment would swap
  in a tuned library behind the same `BaseModel` interface.
- **Full-data refit per trial.** Each trial's shipped model is refit on
  all training data after CV scoring, so the champion is trained on
  strictly more data than any fold model. The trade-off is that the
  reported CV score is slightly pessimistic relative to the shipped
  model — the honest direction.
- **Simplicity tie-break.** When scores tie, the simpler model wins.
  This encodes a real production preference: a stump you can explain
  beats a logistic model you cannot, at equal quality.
- **Slice gates, not just global metrics.** Averages hide subgroup
  failure. The gate is deliberately strict: one failing slice blocks
  promotion, and the failure reason is recorded.
- **JSON registry.** Parameters (weights, tree structure, k-NN training
  set) serialize to JSON, so champions are inspectable and versionable
  in git. The cost is size for k-NN, which stores its training data —
  acceptable at this scale, and a documented reason to prefer
  parametric models for large deployments.
- **Optional FastAPI.** The service module is the only place FastAPI is
  imported, so the library never breaks on a minimal install. Tests
  skip gracefully via `pytest.importorskip` when FastAPI is absent.

## What it demonstrates

Deterministic ML experimentation (seeded splits, reproducible
rankings), honest model selection (CV + baseline margin + slice
checks), a promotion workflow with an auditable gate, a serializable
model registry, and a clean separation between the dependency-free
core and the optional serving layer — the shape of a real enterprise
ML platform, at portfolio scale.
