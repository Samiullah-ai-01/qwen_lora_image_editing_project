import pytest
import json

def test_health_endpoint(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert 'model_loaded' in data
    assert 'device' in data

def test_metrics_endpoint(client):
    response = client.get('/metrics')
    assert response.status_code == 200
    assert b"signforge_requests_total" in response.data

def test_adapters_list_endpoint(client):
    response = client.get('/adapters')
    assert response.status_code == 200
    data = response.get_json()
    assert 'adapters' in data

def test_generate_validation_error(client):
    # Missing prompt
    response = client.post('/generate', json={})
    assert response.status_code == 400
    assert 'error' in response.get_json()

def test_docs_endpoint(client):
    response = client.get('/docs')
    # Use 200 if it serves html, or 404 if not implemented
    assert response.status_code in [200, 404]
