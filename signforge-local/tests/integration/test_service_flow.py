import pytest
import time
from unittest.mock import MagicMock, patch
from signforge.inference.service import InferenceService
from PIL import Image

@pytest.fixture
def mock_pipeline():
    with patch('signforge.inference.service.get_pipeline') as mock:
        pipeline = MagicMock()
        pipeline.is_loaded = True
        pipeline.is_loading = False
        # Mock generate return
        mock_result = MagicMock()
        mock_result.image = Image.new('RGB', (100, 100))
        mock_result.to_dict.return_value = {"status": "ok"}
        pipeline.generate.return_value = mock_result
        mock.return_value = pipeline
        yield pipeline

def test_service_submission_and_processing(mock_pipeline, test_dir):
    service = InferenceService()
    # Mock output dir to use test_dir
    service._output_dir = test_dir / "runs"
    service._output_dir.mkdir(parents=True, exist_ok=True)
    
    service.start(load_model=False) # Don't load real model
    
    request_data = {
        "prompt": "Test signboard",
        "width": 1024,
        "height": 768
    }
    
    # Submit job
    submit_res = service.submit(request_data)
    item_id = submit_res["item_id"]
    assert item_id is not None
    
    # Wait for processing (it's threaded)
    max_wait = 5
    start_time = time.time()
    completed = False
    
    while time.time() - start_time < max_wait:
        status = service.get_status(item_id)
        if status and status["status"] == "completed":
            completed = True
            break
        time.sleep(0.5)
    
    assert completed, f"Job {item_id} did not complete in time"
    
    # Check result
    result = service.get_result(item_id)
    assert result is not None
    assert "image_url" in result
    
    service.stop()
