from pathlib import Path
import sys

import pandas as pd


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate import evaluate_classification
from train import get_cv_splits


def test_evaluate_classification_returns_core_metrics():
    metrics = evaluate_classification(
        y_true=[0, 1, 1, 0],
        y_pred=[0, 1, 0, 0],
        y_score=[0.1, 0.8, 0.4, 0.3],
    )

    assert metrics["accuracy"] == 0.75
    assert "roc_auc" in metrics
    assert metrics["confusion_matrix"] == [[2, 0], [1, 1]]
    assert "macro avg" in metrics["classification_report"]


def test_get_cv_splits_caps_at_smallest_class_size():
    y = pd.Series([0, 0, 0, 1, 1, 1, 1])
    assert get_cv_splits(y, max_splits=5) == 3


def test_get_cv_splits_never_drops_below_two():
    y = pd.Series([0, 1])
    assert get_cv_splits(y, max_splits=5) == 2
