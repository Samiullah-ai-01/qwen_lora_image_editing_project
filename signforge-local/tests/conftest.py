import os
import shutil
import pytest
from pathlib import Path
from flask.testing import FlaskClient

from signforge.core.config import AppConfig, get_config
from signforge.server.app import create_app

@pytest.fixture(scope="session")
def test_dir(tmp_path_factory):
    """Create a temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("signforge_test")

@pytest.fixture(scope="session")
def mock_config(test_dir):
    """Create a mock configuration."""
    config_path = test_dir / "test_config.yaml"
    
    # Create necessary directories
    (test_dir / "models" / "loras").mkdir(parents=True)
    (test_dir / "outputs").mkdir(parents=True)
    (test_dir / "logs").mkdir(parents=True)
    
    # Set env vars to point to test dir
    os.environ["SIGNFORGE_CONFIG_DIR"] = str(test_dir)
    os.environ["SIGNFORGE_MOCK"] = "1"
    
    return get_config()

@pytest.fixture
def app(mock_config):
    """Create a Flask app for testing."""
    app = create_app(test_mode=True)
    app.config.update({
        "TESTING": True,
    })
    return app

@pytest.fixture
def client(app) -> FlaskClient:
    """Create a test client."""
    return app.test_client()

@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after tests."""
    yield
    if "SIGNFORGE_MOCK" in os.environ:
        del os.environ["SIGNFORGE_MOCK"]
