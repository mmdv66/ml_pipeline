import logging
from typing import Optional

import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

logger = logging.getLogger(__name__)


class FeaturePreprocessor:
    def __init__(
        self,
        numeric_features: Optional[list[str]] = None,
        categorical_features: Optional[list[str]] = None,
        encoder_cfg: Optional[DictConfig] = None,
    ) -> None:
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self._encoder_cfg = encoder_cfg
        self._pipeline: Optional[ColumnTransformer] = None

    def _build_encoder(self) -> TransformerMixin:
        if self._encoder_cfg is not None:
            encoder = instantiate(self._encoder_cfg)
            logger.info("Using encoder: %s", self._encoder_cfg.get("_target_", "unknown"))
            return encoder
        return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def build(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        if self.numeric_features is None:
            self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        if self.categorical_features is None:
            self.categorical_features = X.select_dtypes(include="object").columns.tolist()

        logger.info(
            "Features: %d numeric, %d categorical",
            len(self.numeric_features),
            len(self.categorical_features),
        )

        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", self._build_encoder()),
        ])

        transformers: list = []
        if self.numeric_features:
            transformers.append(("num", numeric_pipe, self.numeric_features))
        if self.categorical_features:
            transformers.append(("cat", categorical_pipe, self.categorical_features))

        self._pipeline = ColumnTransformer(transformers=transformers, remainder="drop")
        return self

    def fit_transform(self, X: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        assert self._pipeline is not None, "Call build() first"
        return self._pipeline.fit_transform(X, y)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        assert self._pipeline is not None, "Call build() first"
        return self._pipeline.transform(X)
