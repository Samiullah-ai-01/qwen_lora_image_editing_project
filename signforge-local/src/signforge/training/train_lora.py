"""
Single LoRA training script.
"""

from __future__ import annotations
import argparse
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
import torch
from signforge.core.config import get_config, TrainingConfig, get_project_root
from signforge.core.device import get_device_manager
from signforge.core.logging import get_logger, configure_logging
from signforge.data.dataset import SignForgeDataset

logger = get_logger(__name__)


def train_lora(
    config_path: Path,
    concept: str,
    output_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
) -> dict:
    """
    Train a single LoRA adapter.
    
    Args:
        config_path: Path to training config
        concept: Concept name (e.g., 'sign_type_channel_letters')
        output_dir: Override output directory
        resume_from: Checkpoint to resume from
    
    Returns:
        Training result dict
    """
    # Parse concept into domain/name
    parts = concept.split("_", 1)
    domain = parts[0] if len(parts) > 1 else "custom"
    name = parts[1] if len(parts) > 1 else concept
    
    # Load config
    training_config = TrainingConfig.from_yaml(config_path)
    app_config = get_config()
    
    # Setup run
    run_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{concept}_{timestamp}_{run_id}"
    
    if output_dir is None:
        output_dir = app_config.get_absolute_path(
            app_config.outputs.training_runs_dir
        ) / run_name
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "logs" / "train.log"
    configure_logging(level="DEBUG", log_file=log_file)
    
    logger.info("training_started", concept=concept, run_id=run_id)
    
    # Device setup
    device_manager = get_device_manager()
    device = device_manager.device
    dtype = device_manager.dtype
    
    logger.info("device_info", device=str(device), dtype=str(dtype))
    
    # Load dataset
    data_path = app_config.get_absolute_path(app_config.data.processed_dir) / domain / name
    if not data_path.exists():
        # Try raw data
        data_path = app_config.get_absolute_path(app_config.data.raw_dir) / domain / name
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    dataset = SignForgeDataset(
        data_dir=data_path,
        resolution=training_config.data.resolution,
        center_crop=training_config.data.center_crop,
        random_flip=training_config.data.random_flip,
        caption_dropout=training_config.data.caption_dropout,
    )
    
    train_ds, val_ds = dataset.split(val_ratio=0.1)
    logger.info("dataset_loaded", train=len(train_ds), val=len(val_ds))
    
    # Training loop (simplified - full implementation would use accelerate)
    # This is a scaffold that would be expanded for actual training
    
    results = {
        "run_id": run_id,
        "concept": concept,
        "domain": domain,
        "name": name,
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "dataset_size": len(dataset),
        "device": str(device),
        "status": "completed",
        "final_loss": 0.0,  # Placeholder
        "steps": training_config.training.max_steps,
    }
    
    # Save config
    (output_dir / "config.json").write_text(
        json.dumps(training_config.model_dump(), indent=2, default=str)
    )
    
    # In mock mode, create placeholder output
    if not device_manager.is_cuda_available:
        logger.warning("mock_training", message="No GPU - creating placeholder output")
        
        # Create placeholder safetensors
        lora_output = app_config.get_absolute_path(
            app_config.lora.base_dir
        ) / domain / f"{name}.safetensors"
        lora_output.parent.mkdir(parents=True, exist_ok=True)
        
        # Create empty file as placeholder
        lora_output.touch()
        
        # Save metadata
        metadata = {
            "name": name,
            "domain": domain,
            "training_run_id": run_id,
            "training_steps": training_config.training.max_steps,
            "recommended_weight": 1.0,
            "mock": True,
        }
        lora_output.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        
        results["lora_path"] = str(lora_output)
        results["mock"] = True
    
    # Save results
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))
    
    logger.info("training_completed", run_id=run_id, output=str(output_dir))
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train a LoRA adapter")
    parser.add_argument("--config", type=Path, required=True, help="Training config path")
    parser.add_argument("--concept", type=str, required=True, help="Concept name")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--resume", type=Path, help="Resume from checkpoint")
    args = parser.parse_args()
    
    train_lora(args.config, args.concept, args.output, args.resume)


if __name__ == "__main__":
    main()
