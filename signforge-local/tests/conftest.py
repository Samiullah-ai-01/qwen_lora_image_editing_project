import os
import pytest
from pathlib import Path
from flask.testing import FlaskClient
from signforge.core.config import AppConfig, get_config
from signforge.server.app import create_app
from unittest.mock import patch, MagicMock

@pytest.fixture(scope="session", autouse=True)
def mock_env():
    """Set up mock environment variables for all tests."""
    os.environ["SIGNFORGE_MOCK"] = "1"
    yield
    if "SIGNFORGE_MOCK" in os.environ:
        del os.environ["SIGNFORGE_MOCK"]

@pytest.fixture(scope="session")
def test_dir(tmp_path_factory):
    """Create a temporary directory for test artifacts."""
    return tmp_path_factory.mktemp("signforge_test")

@pytest.fixture(autouse=True)
def mock_service_singleton():
    """Ensure get_service always returns a mock for all callers."""
    with patch('signforge.inference.service.get_service') as mock:
        service = MagicMock()
        service._started = True
        service.get_queue_status.return_value = {"queue_size": 0, "max_size": 10, "running": True}
        mock.return_value = service
        yield service

@pytest.fixture
def app():
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
