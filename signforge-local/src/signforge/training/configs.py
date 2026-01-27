"""
Training configuration utilities and constants.
"""

from __future__ import annotations
from typing import Any, Dict

from signforge.core.config import TrainingConfig


def get_default_optimizer_params(config: TrainingConfig) -> Dict[str, Any]:
    """Extract optimizer parameters from config."""
    params = {
        "lr": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
    }
    
    if config.training.optimizer.lower() == "adamw":
        params.update({
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        })
        
    return params


def get_scheduler_params(config: TrainingConfig, total_steps: int) -> Dict[str, Any]:
    """Extract LR scheduler parameters."""
    return {
        "num_warmup_steps": config.training.lr_warmup_steps,
        "num_training_steps": total_steps,
    }
