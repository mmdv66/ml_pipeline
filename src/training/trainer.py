import logging

import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import cross_val_score

from src.models.sklearn_model import SklearnModel

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def cross_validate(self, model: SklearnModel, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        logger.info("%d-fold CV, scoring=%s", self.cfg.cv_folds, self.cfg.scoring)
        scores = cross_val_score(
            model.estimator, X, y,
            cv=self.cfg.cv_folds,
            scoring=self.cfg.scoring,
            n_jobs=-1,
        )
        results = {
            f"cv_{self.cfg.scoring}_mean": float(scores.mean()),
            f"cv_{self.cfg.scoring}_std": float(scores.std()),
        }
        logger.info("CV %s: %.4f ± %.4f", self.cfg.scoring, scores.mean(), scores.std())
        return results

    def fit(self, model: SklearnModel, X_train: np.ndarray, y_train: np.ndarray) -> SklearnModel:
        return model.fit(X_train, y_train)
