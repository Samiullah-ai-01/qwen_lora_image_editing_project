"""
Configuration management for SignForge.

Loads and validates YAML configuration files with override support.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


def get_project_root() -> Path:
    """Get the project root directory."""
    # Start from this file and go up until we find pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Fallback to current working directory
    return Path.cwd()


class ModelConfig(BaseModel):
    """Model configuration."""
    
    base_path: str = "models/base/Qwen-Image"
    hf_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    dtype: str = "bf16"
    use_xformers: bool = True
    attention_slicing: str = "auto"
    vae_tiling: bool = False
    use_tiny_vae: bool = False
    enable_turbo: bool = False
    cpu_offload: bool = False


class InferenceConfig(BaseModel):
    """Inference configuration."""
    
    max_queue_size: int = 10
    timeout_seconds: int = 120
    default_steps: int = 30
    default_guidance_scale: float = 7.5
    default_width: int = 1024
    default_height: int = 768
    scheduler: str = "euler_a"
    max_resolution: int = 2097152
    default_seed: int = -1


class LoRAConfig(BaseModel):
    """LoRA adapter configuration."""
    
    base_dir: str = "models/loras"
    cache_dir: str = "models/adapters_cache"
    max_cached: int = 10
    default_weights: dict[str, float] = Field(default_factory=lambda: {
        "sign_type": 1.0,
        "mounting": 0.9,
        "perspective": 0.7,
        "environment": 0.9,
        "lighting": 0.8,
        "material": 0.8,
    })
    normalize_weights: bool = True


class ServerConfig(BaseModel):
    """Server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 5000
    cors_enabled: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:5173",
        "http://localhost:3000",
    ])
    static_dir: str = "src/signforge/server/static"
    max_content_length: int = 10485760
    rate_limit_enabled: bool = True
    rate_limit_per_minute: int = 30


class DataConfig(BaseModel):
    """Data directories configuration."""
    
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    splits_dir: str = "data/splits"
    eval_dir: str = "data/eval_sets"
    logos_dir: str = "data/logos"
    backgrounds_dir: str = "data/backgrounds"


class OutputsConfig(BaseModel):
    """Output directories configuration."""
    
    training_runs_dir: str = "outputs/training_runs"
    inference_runs_dir: str = "outputs/inference_runs"


class MonitoringConfig(BaseModel):
    """Monitoring configuration."""
    
    enabled: bool = True
    prometheus_port: int = 9100
    track_latency: bool = True
    track_queue_depth: bool = True
    track_gpu_memory: bool = True
    track_adapter_usage: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    format: str = "json"
    level: str = "INFO"
    file: str = "logs/app.log"
    rotate: bool = True
    max_bytes: int = 10485760
    backup_count: int = 5


class SafetyConfig(BaseModel):
    """Safety and validation configuration."""
    
    max_prompt_length: int = 1000
    blocked_words: list[str] = Field(default_factory=list)
    validate_images: bool = True
    allowed_image_formats: list[str] = Field(default_factory=lambda: [
        "png", "jpg", "jpeg", "webp"
    ])
    max_upload_width: int = 4096
    max_upload_height: int = 4096


class AppInfo(BaseModel):
    """Application info."""
    
    name: str = "signforge-local"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"
    batch_size: int = 1


class AppConfig(BaseModel):
    """Main application configuration."""
    
    app: AppInfo = Field(default_factory=AppInfo)
    model: ModelConfig = Field(default_factory=ModelConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    outputs: OutputsConfig = Field(default_factory=OutputsConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to a YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Convert a config-relative path to absolute."""
        root = get_project_root()
        return root / relative_path


class TrainingLoRAConfig(BaseModel):
    """LoRA-specific training parameters."""
    
    rank: int = 32
    alpha: int = 64
    dropout: float = 0.1
    target_modules: list[str] = Field(default_factory=lambda: [
        "to_q", "to_k", "to_v", "to_out.0", "proj_in", "proj_out"
    ])
    init_weights: str = "gaussian"


class TrainingDataConfig(BaseModel):
    """Training data configuration."""
    
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = True
    caption_dropout: float = 0.1
    num_workers: int = 4
    shuffle: bool = True


class TrainingValidationConfig(BaseModel):
    """Validation configuration for training."""
    
    validate_every: int = 200
    num_samples: int = 4
    prompts: list[str] = Field(default_factory=lambda: [
        "A professional channel letter sign reading 'CAFE' on a brick wall",
        "Modern LED signage for a tech company",
        "Vintage neon sign at night",
        "Corporate building entrance sign",
    ])
    seed: int = 42


class TrainingCheckpointConfig(BaseModel):
    """Checkpointing configuration."""
    
    save_every: int = 200
    keep_last: int = 3
    save_optimizer: bool = True


class TrainingExportConfig(BaseModel):
    """Export configuration."""
    
    format: str = "safetensors"
    include_metadata: bool = True
    metadata: list[str] = Field(default_factory=lambda: [
        "training_config", "dataset_hash", "final_loss", "training_steps"
    ])


class TrainingHardwareConfig(BaseModel):
    """Hardware configuration for training."""
    
    device: str = "auto"
    num_gpus: int = 1


class TrainingParams(BaseModel):
    """Training parameters."""
    
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_steps: int = 1000
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    mixed_precision: str = "bf16"
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42


class TrainingConfig(BaseModel):
    """Complete training configuration."""
    
    training: TrainingParams = Field(default_factory=TrainingParams)
    lora: TrainingLoRAConfig = Field(default_factory=TrainingLoRAConfig)
    data: TrainingDataConfig = Field(default_factory=TrainingDataConfig)
    validation: TrainingValidationConfig = Field(default_factory=TrainingValidationConfig)
    checkpointing: TrainingCheckpointConfig = Field(default_factory=TrainingCheckpointConfig)
    export: TrainingExportConfig = Field(default_factory=TrainingExportConfig)
    hardware: TrainingHardwareConfig = Field(default_factory=TrainingHardwareConfig)
    
    @classmethod
    def from_yaml(cls, path: Path) -> "TrainingConfig":
        """Load from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


class InferenceProfileConfig(BaseModel):
    """Inference profile configuration."""
    
    steps: int = 30
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 768
    scheduler: str = "euler_a"
    seed: int = -1
    
    @classmethod
    def from_yaml(cls, path: Path) -> "InferenceProfileConfig":
        """Load from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        # Handle nested 'generation' key
        if "generation" in data:
            data = data["generation"]
        return cls(**data)


_config: Optional[AppConfig] = None


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load the application configuration.
    
    Args:
        config_path: Optional path to config file. Defaults to configs/app.yaml
        
    Returns:
        Loaded configuration
    """
    global _config
    
    if config_path is None:
        config_path = get_project_root() / "configs" / "app.yaml"
    
    if config_path.exists():
        _config = AppConfig.from_yaml(config_path)
    else:
        # Use defaults
        _config = AppConfig()
    
    return _config


def get_config() -> AppConfig:
    """Get the current configuration (loads if not already loaded)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def validate_all_configs() -> dict[str, bool]:
    """
    Validate all configuration files in the configs directory.
    
    Returns:
        Dict mapping config file paths to validation status
    """
    root = get_project_root()
    configs_dir = root / "configs"
    results = {}
    
    # Main app config
    app_config = configs_dir / "app.yaml"
    if app_config.exists():
        try:
            AppConfig.from_yaml(app_config)
            results[str(app_config)] = True
        except Exception as e:
            results[str(app_config)] = False
            print(f"Error in {app_config}: {e}")
    
    # Training configs
    training_dir = configs_dir / "training"
    if training_dir.exists():
        for config_file in training_dir.glob("*.yaml"):
            try:
                TrainingConfig.from_yaml(config_file)
                results[str(config_file)] = True
            except Exception as e:
                results[str(config_file)] = False
                print(f"Error in {config_file}: {e}")
    
    # Inference configs
    inference_dir = configs_dir / "inference"
    if inference_dir.exists():
        for config_file in inference_dir.glob("**/*.yaml"):
            try:
                InferenceProfileConfig.from_yaml(config_file)
                results[str(config_file)] = True
            except Exception as e:
                results[str(config_file)] = False
                print(f"Error in {config_file}: {e}")
    
    return results
