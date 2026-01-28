import torch
from signforge.core.device import get_device_manager

def test_device_detection():
    dm = get_device_manager()
    assert dm.device is not None
    # On most dev machines without GPU, this should be cpu
    if not torch.cuda.is_available():
        assert str(dm.device) == "cpu"

def test_memory_info_format():
    dm = get_device_manager()
    info = dm.get_memory_info()
    assert "device" in info
    if info.get("available"):
        assert "total_gb" in info

def test_recommended_settings():
    dm = get_device_manager()
    settings = dm.get_recommended_settings()
    assert "attention_slicing" in settings
    assert "vae_tiling" in settings
