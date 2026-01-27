"""
Checkpoint management for SignForge training.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from signforge.core.logging import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """Handles saving and loading of model checkpoints and Adapters."""
    
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        is_best: bool = False,
    ) -> Path:
        """Save a training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint-step-{step}.pt"
        
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        
        torch.save(state, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(state, best_path)
            
        logger.info("checkpoint_saved", path=str(checkpoint_path), step=step)
        return checkpoint_path

    def save_lora_adapter(
        self,
        adapter_state_dict: Dict[str, torch.Tensor],
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save the LoRA weights specifically (e.g. as .safetensors)."""
        from safetensors.torch import save_file
        
        adapter_path = self.output_dir / f"{name}.safetensors"
        save_file(adapter_state_dict, adapter_path, metadata=metadata)
        
        logger.info("lora_adapter_saved", path=str(adapter_path))
        return adapter_path

    def load_checkpoint(
        self,
        checkpoint_path: Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, Any]:
        """Load a training checkpoint."""
        logger.info("loading_checkpoint", path=str(checkpoint_path))
        state = torch.load(checkpoint_path, map_location="cpu")
        
        model.load_state_dict(state["model_state_dict"])
        if optimizer and "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"])
            
        return state

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the most recent checkpoint in the directory."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint-step-*.pt"))
        if not checkpoints:
            return None
            
        # Extract step numbers and sort
        checkpoints.sort(key=lambda x: int(os.path.basename(x).split("-")[2].split(".")[0]))
        return checkpoints[-1]
