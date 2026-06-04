import logging

import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class SklearnModel:
    def __init__(self, cfg: DictConfig) -> None:
        from omegaconf import OmegaConf
        self.name: str = cfg.name
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        cfg_dict.pop("name", None)
        self._model: BaseEstimator = instantiate(OmegaConf.create(cfg_dict))
        logger.info("Initialized model: %s", self.name)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SklearnModel":
        logger.info("Fitting %s on %d samples", self.name, X.shape[0])
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def estimator(self) -> BaseEstimator:
        return self._model
