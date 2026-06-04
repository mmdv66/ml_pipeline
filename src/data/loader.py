import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, path: str, target_col: str, test_size: float, random_state: int) -> None:
        self.path = Path(path)
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        logger.info("Loading data from %s", self.path)
        df = pd.read_csv(self.path)
        logger.info("Loaded %d rows, %d columns", len(df), df.shape[1])

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info("Split: train=%d, test=%d", len(X_train), len(X_test))
        return X_train, X_test, y_train, y_test
