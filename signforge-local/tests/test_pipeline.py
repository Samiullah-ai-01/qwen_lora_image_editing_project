import pytest
from signforge.ml.pipeline import get_pipeline, MockPipeline
from signforge.ml.prompt.templates import get_template

def test_mock_pipeline_initialization():
    """Test that mock pipeline initializes correctly."""
    pipeline = get_pipeline()
    assert pipeline is not None
    # In test env, it might be the real one or mock depending on setup, 
    # but we can check if it adheres to the interface

def test_prompt_templates():
    """Test prompt template rendering."""
    template = get_template("channel_letters")
    assert template is not None
    
    prompt = template.render(
        text="TEST",
        material="metal",
        lighting="LED",
        surface="brick",
        perspective="front",
        environment="city"
    )
    
    assert "TEST" in prompt
    assert "metal" in prompt
    assert "LED" in prompt

def test_mock_generation():
    """Test generation with mock pipeline."""
    pipeline = MockPipeline()
    pipeline.load()
    
    from signforge.ml.pipeline import GenerationRequest
    req = GenerationRequest(
        prompt="test prompt",
        width=512,
        height=512,
        steps=1
    )
    
    result = pipeline.generate(req)
    assert result.image is not None
    assert result.image.size == (512, 512)
    assert result.seed != -1
