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
class ModelConfig:
    baseline: BaselineConfig
    mlp: MLPConfig

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

        return cls(
            dataset=raw.get("DATASET", "mlg-ulb/creditcardfraud"),
            output_dir=Path(raw.get("OUTPUT_DIR", "data/raw")),
            models=ModelConfig(
                baseline=BaselineConfig(
                    max_iter=models_cfg.get("baseline", {}).get("max_iter", 1000),
                    class_weight=models_cfg.get("baseline", {}).get("class_weight", "balanced"),
                ),
                mlp=MLPConfig(
                    hidden_size=models_cfg.get("mlp", {}).get("hidden_size", 64),
                    output_size=models_cfg.get("mlp", {}).get("output_size", 1),
                    epochs=models_cfg.get("mlp", {}).get("epochs", 10),
                    batch_size=models_cfg.get("mlp", {}).get("batch_size", 256),
                    lr=models_cfg.get("mlp", {}).get("lr", 0.001),
                    dropout=models_cfg.get("mlp", {}).get("dropout", 0.2),
                    threshold=models_cfg.get("mlp", {}).get("threshold", 0.5)

                ),
            ),
            training=TrainingConfig(
                test_size=training_cfg.get("test_size", 0.2),
                random_state=training_cfg.get("random_state", 42),
                stratify=training_cfg.get("stratify", True),
            ),
            env=EnvConfig.from_env(),
        )