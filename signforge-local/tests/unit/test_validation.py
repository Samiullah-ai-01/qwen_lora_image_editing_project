import pytest
from pydantic import ValidationError
from signforge.inference.api_schema import GenerateRequest

def test_valid_generate_request():
    req = GenerateRequest(
        prompt="A sign for a bakery",
        width=1024,
        height=768,
        steps=20
    )
    assert req.prompt == "A sign for a bakery"
    assert req.logo_image_b64 is None

def test_invalid_prompt_request():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="", width=1024)

def test_invalid_resolution_request():
    with pytest.raises(ValidationError):
        GenerateRequest(prompt="Valid", width=50) # Too small

def test_base64_image_fields():
    req = GenerateRequest(
        prompt="Valid",
        logo_image_b64="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    assert req.logo_image_b64 is not None
