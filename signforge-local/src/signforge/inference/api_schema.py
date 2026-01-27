"""
API schema definitions for inference endpoints.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    """Request schema for generation endpoint."""
    prompt: str = Field(..., min_length=1, max_length=1000)
    negative_prompt: str = Field(default="", max_length=500)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=768, ge=256, le=2048)
    steps: int = Field(default=30, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1, le=20)
    seed: int = Field(default=-1)
    adapters: list[str] = Field(default_factory=list)
    adapter_weights: list[float] = Field(default_factory=list)
    normalize_weights: bool = Field(default=True)
    profile: Optional[str] = Field(default=None)

    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "Professional channel letter sign reading 'CAFE'",
                "negative_prompt": "blurry, low quality",
                "width": 1024,
                "height": 768,
                "steps": 30,
                "adapters": ["sign_type/channel_letters"],
                "adapter_weights": [1.0],
            }
        }


class GenerateResponse(BaseModel):
    """Response schema for generation endpoint."""
    item_id: str
    status: str
    message: Optional[str] = None


class StatusResponse(BaseModel):
    """Response for status endpoint."""
    id: str
    status: str
    progress: int = 0
    total_steps: int = 0
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


class ResultResponse(BaseModel):
    """Response for result endpoint."""
    item_id: str
    image_url: str
    image_path: str
    seed: int
    prompt: str
    width: int
    height: int
    steps: int
    adapters: list[str]
    adapter_weights: list[float]
    generation_time_ms: int


class AdapterInfo(BaseModel):
    """Schema for adapter information."""
    name: str
    domain: str
    path: str
    file_size_mb: float
    recommended_weight: float
    training_run_id: Optional[str] = None


class AdaptersResponse(BaseModel):
    """Response for adapters endpoint."""
    domains: list[str]
    adapters: dict[str, list[AdapterInfo]]
    total_count: int


class HealthResponse(BaseModel):
    """Response for health endpoint."""
    status: str
    model_loaded: bool
    is_mock: bool
    device: str
    queue_size: int
    queue_max: int
    gpu_memory_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None


class QueueStatusResponse(BaseModel):
    """Response for queue status."""
    running: bool
    queue_size: int
    max_size: int
    current_item: Optional[str] = None
    total_processed: int
    session_id: str
