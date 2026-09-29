import textwrap

import pytest


def test_settings_from_yaml_parses_feature_selection(tmp_path, monkeypatch):
    from src.config.settings import Settings

    config_content = textwrap.dedent("""
        dataset:
          name: test/dataset
          output_dir: data/test

        training:
          test_size: 0.25
          cv_folds: 3
          scoring: recall

        tuning:
          n_trials: 5

        feature_selection:
          enabled: true
          method: greedy_forward
          output: data/feats.json
          cv_folds: 2
          scoring: pr_auc
          max_features: 10
          min_improvement: 0.001

        models:
          baseline:
            fixed:
              C: 1.0
            search: {}
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DB_USER", "test")
    monkeypatch.setenv("DB_PASSWORD", "test")
    monkeypatch.setenv("DB_NAME", "test")

    settings = Settings.from_yaml(str(config_path))

    assert settings.training.cv_folds == 3
    assert settings.training.scoring == "recall"
    assert settings.feature_selection.enabled is True
    assert settings.feature_selection.max_features == 10
    assert settings.feature_selection.min_improvement == 0.001
    assert "baseline" in settings.models
    assert settings.models["baseline"].fixed["C"] == 1.0


def test_settings_validates_scoring_values(tmp_path, monkeypatch):
    from src.config.settings import Settings

    config_content = textwrap.dedent("""
        feature_selection:
          scoring: invalid_scoring
        models: {}
    """)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    monkeypatch.setenv("DB_USER", "test")
    monkeypatch.setenv("DB_PASSWORD", "test")
    monkeypatch.setenv("DB_NAME", "test")

    with pytest.raises((ValueError, KeyError)):
        Settings.from_yaml(str(config_path))