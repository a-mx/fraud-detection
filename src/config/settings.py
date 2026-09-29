from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.config.env import EnvConfig


@dataclass(frozen=True)
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    cv_folds: int = 5
    scoring: str = "pr_auc"


@dataclass(frozen=True)
class TuningConfig:
    n_trials: int = 30
    timeout: int | None = None
    direction: str = "maximize"


@dataclass(frozen=True)
class ModelSpec:
    fixed: dict[str, Any] = field(default_factory=dict)
    search: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class FeatureSelectionConfig:
    enabled: bool = True
    method: str = "greedy_forward"
    output: str = "data/selected_features.json"
    cv_folds: int = 3
    scoring: str = "pr_auc"
    max_features: int | None = None
    min_improvement: float = 1e-4


@dataclass(frozen=True)
class Settings:
    dataset: str
    output_dir: Path
    training: TrainingConfig
    tuning: TuningConfig
    models: dict[str, ModelSpec]
    feature_selection: FeatureSelectionConfig
    env: EnvConfig

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Settings":
        load_dotenv()
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        ds = raw.get("dataset", {})
        tr = raw.get("training", {})
        tu = raw.get("tuning", {})
        fs = raw.get("feature_selection", {})

        models = {
            name: ModelSpec(
                fixed=spec.get("fixed", {}),
                search=spec.get("search", {}),
            )
            for name, spec in raw.get("models", {}).items()
        }

        return cls(
            dataset=ds.get("name", "mlg-ulb/creditcardfraud"),
            output_dir=Path(ds.get("output_dir", "data/raw")),
            training=TrainingConfig(**tr),
            tuning=TuningConfig(**tu),
            feature_selection = FeatureSelectionConfig(**fs),
            models=models,
            env=EnvConfig.from_env(),
        )