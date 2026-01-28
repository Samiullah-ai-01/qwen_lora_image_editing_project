import pytest
from pathlib import Path
from signforge.core.config import AppConfig, ModelConfig, get_project_root

def test_project_root():
    root = get_project_root()
    assert root.exists()
    assert (root / "src").exists()

def test_default_config():
    config = AppConfig()
    assert config.app.name == "signforge-local"

def test_model_config_assignment():
    # Pydantic validates on initialization
    config = ModelConfig(dtype="fp32")
    assert config.dtype == "fp32"

def test_absolute_path_resolution():
    config = AppConfig()
    rel_path = "models/base"
    abs_path = config.get_absolute_path(rel_path)
    assert abs_path.is_absolute()
    assert str(get_project_root()) in str(abs_path)
