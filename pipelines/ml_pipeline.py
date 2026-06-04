import logging
import sys
from pathlib import Path

import hydra
import mlflow
import mlflow.sklearn
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import FeaturePreprocessor
from src.evaluation.evaluator import Evaluator
from src.models.sklearn_model import SklearnModel
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run_pipeline(cfg: DictConfig) -> float:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.project.experiment_name)

    with mlflow.start_run(run_name=cfg.model.name):
        mlflow.log_params(OmegaConf.to_container(cfg.model, resolve=True))
        mlflow.log_params(OmegaConf.to_container(cfg.training, resolve=True))
        mlflow.log_param("encoder", cfg.encoder.get("_target_", "unknown").split(".")[-1])

        loader = DataLoader(
            path=cfg.data.path,
            target_col=cfg.data.target_col,
            test_size=cfg.data.test_size,
            random_state=cfg.data.random_state,
        )
        X_train, X_test, y_train, y_test = loader.load()

        preprocessor = FeaturePreprocessor(encoder_cfg=cfg.encoder)
        preprocessor.build(X_train)
        X_train_proc = preprocessor.fit_transform(X_train, y_train.values)
        X_test_proc = preprocessor.transform(X_test)

        model = SklearnModel(cfg.model)
        trainer = Trainer(cfg.training)

        cv_metrics = trainer.cross_validate(model, X_train_proc, y_train)
        mlflow.log_metrics(cv_metrics)

        trainer.fit(model, X_train_proc, y_train)

        evaluator = Evaluator()
        test_metrics = evaluator.evaluate(model, X_test_proc, y_test)
        mlflow.log_metrics(test_metrics)

        registered_name = cfg.project.name if cfg.training.register_model else None
        mlflow.sklearn.log_model(
            sk_model=model.estimator,
            artifact_path="model",
            registered_model_name=registered_name,
        )

        roc_auc = test_metrics["roc_auc"]
        logger.info("Pipeline complete. ROC-AUC=%.4f", roc_auc)
        return roc_auc


if __name__ == "__main__":
    run_pipeline()
