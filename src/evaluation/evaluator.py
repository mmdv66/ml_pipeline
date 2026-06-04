import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.sklearn_model import SklearnModel

logger = logging.getLogger(__name__)


class Evaluator:
    def evaluate(self, model: SklearnModel, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics: dict[str, float] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "f1": f1_score(y_test, y_pred, average="binary"),
            "precision": precision_score(y_test, y_pred, average="binary"),
            "recall": recall_score(y_test, y_pred, average="binary"),
        }

        logger.info("\n%s", classification_report(y_test, y_pred))
        for name, value in metrics.items():
            logger.info("  %-12s %.4f", name, value)

        return metrics
