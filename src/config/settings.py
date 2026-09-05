# src/config/settings.py
from dataclasses import dataclass
from pathlib import Path
import yaml
from dotenv import load_dotenv
from src.config.env import EnvConfig

@dataclass(frozen=True)
class MLPConfig:
    hidden_size: int
    output_size: int
    epochs: int
    batch_size: int
    lr: float
    dropout: float
    threshold: float

@dataclass(frozen=True)
class BaselineConfig:
    max_iter: int
    class_weight: str

@dataclass(frozen=True)
class XGBConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    reg_alpha: float
    reg_lambda: float
    n_jobs: int
    random_state: int

@dataclass(frozen=True)
class ModelConfig:
    baseline: BaselineConfig
    mlp: MLPConfig
    xgb: XGBConfig

@dataclass(frozen=True)
class TrainingConfig:
    test_size: float
    random_state: int
    stratify: bool

@dataclass(frozen=True)
class Settings:
    dataset: str
    output_dir: Path
    models: ModelConfig
    training: TrainingConfig
    env: EnvConfig

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Settings":
        load_dotenv()
        
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        models_cfg = raw.get("models", {})
        training_cfg = raw.get("training", {})

        baseline_cfg = models_cfg.get("baseline", {})
        mlp_cfg = models_cfg.get("mlp", {})
        xgb_cfg = models_cfg.get("xgb", {})

        return cls(
            dataset=raw.get("DATASET", "mlg-ulb/creditcardfraud"),
            output_dir=Path(raw.get("OUTPUT_DIR", "data/raw")),
            models=ModelConfig(
                baseline=BaselineConfig(
                    max_iter=baseline_cfg.get("max_iter", 1000),
                    class_weight=baseline_cfg.get("class_weight", "balanced"),
                ),
                mlp=MLPConfig(
                    hidden_size=mlp_cfg.get("hidden_size", 64),
                    output_size=mlp_cfg.get("output_size", 1),
                    epochs=mlp_cfg.get("epochs", 10),
                    batch_size=mlp_cfg.get("batch_size", 256),
                    lr=mlp_cfg.get("lr", 0.001),
                    dropout=mlp_cfg.get("dropout", 0.2),
                    threshold=mlp_cfg.get("threshold", 0.5)

                ),
                xgb=XGBConfig(
                    n_estimators=xgb_cfg.get("n_estimators", 100),
                    max_depth=xgb_cfg.get("max_depth", 6),
                    learning_rate=xgb_cfg.get("learning_rate", 0.1),
                    subsample=xgb_cfg.get("subsample", 0.8),
                    colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
                    min_child_weight=xgb_cfg.get("min_child_weight", 1),
                    reg_alpha=xgb_cfg.get("reg_alpha", 0.0),
                    reg_lambda=xgb_cfg.get("reg_lambda", 1.0),
                    n_jobs=xgb_cfg.get("n_jobs", -1),
                    random_state=xgb_cfg.get("random_state", 42),
                ),
            ),
            training=TrainingConfig(
                test_size=training_cfg.get("test_size", 0.2),
                random_state=training_cfg.get("random_state", 42),
                stratify=training_cfg.get("stratify", True),
            ),
            env=EnvConfig.from_env(),
        )