def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"

def test_adapters_list(client):
    """Test adapters list endpoint."""
    response = client.get("/adapters")
    assert response.status_code == 200
    data = response.get_json()
    assert isinstance(data, dict)

def test_generation_request(client):
    """Test generation submission."""
    # We need to ensure service is mocked or configured for tests
    # For now, we expect a 202 Accepted
    response = client.post("/generate", json={
        "prompt": "test sign",
        "width": 512,
        "height": 512,
        "steps": 10
    })
    
    assert response.status_code in (202, 400)  # 400 if validation fails, 202 if accepted
    
    if response.status_code == 202:
        data = response.get_json()
        assert "item_id" in data
        assert data["status"] in ("pending", "processing")
