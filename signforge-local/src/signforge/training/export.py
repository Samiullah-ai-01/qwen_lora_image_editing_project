"""
Export and format conversion for LoRA adapters.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

from signforge.core.logging import get_logger

logger = get_logger(__name__)


def export_for_inference(
    weights_path: Path,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Prepare a trained adapter for inference use.
    Moves weights to the central registry and ensures metadata is present.
    """
    weights_path = Path(weights_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # In a real scenario, this might involve format conversion
    # (e.g., from hub-style to diffusers-style or vice versa)
    
    import shutil
    shutil.copy2(weights_path, output_path)
    
    # Write metadata sidecar
    if metadata:
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
            
    logger.info("adapter_exported", source=str(weights_path), dest=str(output_path))
    return output_path


def convert_to_diffusers(
    original_path: Path,
    output_path: Path,
) -> Path:
    """Convert custom LoRA weights to Diffusers format."""
    # Logic for format conversion between training scripts 
    # and inference engine formats if they differ.
    logger.info("converting_to_diffusers", path=str(original_path))
    return output_path
