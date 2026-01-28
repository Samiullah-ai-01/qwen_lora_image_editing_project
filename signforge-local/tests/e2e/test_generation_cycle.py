import pytest
import time
from unittest.mock import patch, MagicMock
from PIL import Image

@pytest.mark.e2e
def test_full_generation_lifecycle(client, mock_service_singleton):
    """Test the full lifecycle from submission to result retrieval."""
    
    # 1. Mock the pipeline so we don't actually run SDXL on CPU
    with patch('signforge.inference.service.get_pipeline') as mock_get_pipe:
        mock_pipe = MagicMock()
        mock_pipe.is_loaded = True
        
        # Mock successful generation
        mock_result = MagicMock()
        mock_result.image = Image.new('RGB', (512, 512))
        mock_result.to_dict.return_value = {
            "prompt": "Test signboard",
            "seed": 123,
            "width": 512,
            "height": 512,
            "steps": 1,
            "generation_time_ms": 100
        }
        mock_pipe.generate.return_value = mock_result
        mock_get_pipe.return_value = mock_pipe

        # 2. Setup mock_service behavior
        item_id = "test-id-123"
        mock_service_singleton.submit.return_value = {"item_id": item_id, "status": "pending"}
        mock_service_singleton.get_status.return_value = {"id": item_id, "status": "completed"}
        mock_service_singleton.get_result.return_value = {
            "item_id": item_id, 
            "image_url": "/fake/path.png",
            "prompt": "Test signboard e2e"
        }

        # 3. Submit a request
        payload = {
            "prompt": "Test signboard e2e",
            "width": 1024,
            "height": 768,
            "steps": 20
        }
        resp = client.post('/generate', json=payload)
        assert resp.status_code == 202
        assert resp.get_json()['item_id'] == item_id
        
        # 4. Poll for status
        s_resp = client.get(f'/generate/{item_id}')
        assert s_resp.status_code == 200
        assert s_resp.get_json()['status'] == 'completed'
        
        # 5. Get result
        r_resp = client.get(f'/generate/{item_id}/result')
        assert r_resp.status_code == 200
        result_data = r_resp.get_json()
        assert result_data['item_id'] == item_id
        assert 'image_url' in result_data
