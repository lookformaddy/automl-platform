"""Prediction service (FastAPI, optional dependency).

The core package never imports FastAPI; only this module does, so the
library, search, and registry all work on plain system Python. The app
serves whatever champion the caller registers via ``set_champion`` — in
production that would be loaded from the registry at startup.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from automl.models import BaseModel

_champion: Optional[BaseModel] = None
_champion_name: str = "none"


def set_champion(model: BaseModel, name: str = "champion") -> None:
    global _champion, _champion_name
    _champion = model
    _champion_name = name


def create_app():  # type: ignore[no-untyped-def]
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel as PydanticModel

    class PredictRequest(PydanticModel):
        features: List[List[float]]

    app = FastAPI(title="automl-platform prediction service")

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok", "champion": _champion_name}

    @app.post("/predict")
    def predict(req: PredictRequest) -> Dict[str, Any]:
        if _champion is None:
            raise HTTPException(status_code=503, detail="no champion registered")
        probas = _champion.predict_proba(req.features)
        return {
            "champion": _champion_name,
            "predictions": [1 if p >= 0.5 else 0 for p in probas],
            "probabilities": probas,
        }

    return app
