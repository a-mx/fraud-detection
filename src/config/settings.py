from dataclasses import dataclass
from pathlib import Path
import yaml

from src.config.env import EnvConfig

@dataclass(frozen=True)
class Settings:
    dataset: str
    output_dir: Path
    env: EnvConfig

    @classmethod
    def from_yaml(cls, config_path: str = "config.yaml") -> "Settings":
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        return cls(
            dataset=raw.get("DATASET", "mlg-ulb/creditcardfraud"),
            output_dir=Path(raw.get("OUTPUT_DIR", "data/raw")),
            env=EnvConfig.from_env(),
        )