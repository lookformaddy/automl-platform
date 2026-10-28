"""Entrypoint for the prediction service: ``uvicorn app:app``.

Requires the optional FastAPI dependency (see requirements.txt).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from automl.service import create_app  # noqa: E402

app = create_app()
