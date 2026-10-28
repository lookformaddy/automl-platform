"""Tabular AutoML platform: deterministic candidate search, leaderboard,
champion promotion gates, and a model registry. Pure standard library core."""

from automl.datasets import SyntheticConfig, generate_classification
from automl.evaluation import accuracy, f1_score, log_loss, roc_auc
from automl.leaderboard import Leaderboard
from automl.models import (
    DecisionTreeClassifier,
    KNeighborsClassifier,
    LogisticRegression,
    MajorityClassifier,
)
from automl.registry import PromotionGate, Registry
from automl.search import SearchConfig, TrialRunner, default_search_space

__all__ = [
    "SyntheticConfig",
    "generate_classification",
    "accuracy",
    "f1_score",
    "log_loss",
    "roc_auc",
    "Leaderboard",
    "DecisionTreeClassifier",
    "KNeighborsClassifier",
    "LogisticRegression",
    "MajorityClassifier",
    "PromotionGate",
    "Registry",
    "SearchConfig",
    "TrialRunner",
    "default_search_space",
]
