def test_version():
    """Test package version."""
    import signforge
    assert signforge.__version__

def test_imports():
    """Test module imports."""
    from signforge.core.config import get_config
    from signforge.ml.pipeline import get_pipeline
    from signforge.server.app import create_app
    
    assert get_config
    assert get_pipeline
    assert create_app

def test_config_loading():
    """Test configuration loading."""
    from signforge.core.config import get_config
    config = get_config()
    assert config.app.name == "signforge-local"
    assert config.app.batch_size > 0
