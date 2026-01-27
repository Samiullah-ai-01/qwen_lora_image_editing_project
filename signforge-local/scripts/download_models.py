#!/usr/bin/env python3
"""
Model download script for SignForge.
Handles download of base model and LoRA adapters.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, hf_hub_download
from tqdm import tqdm

# Constants
DEFAULT_MODEL_ID = "Qwen/Qwen-VL-Chat"  # Using Qwen-VL as base for now, can be swapped
# In a real scenario, this would be the specific Qwen-Image model ID
MODEL_CACHE_DIR = "models/base"
LORA_CACHE_DIR = "models/loras"


def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_model(
    repo_id: str,
    local_dir: Path,
    token: Optional[str] = None,
    resume: bool = True,
) -> None:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        repo_id: HF repository ID
        local_dir: Local directory to save model
        token: HF API token
        resume: Resume download if interrupted
    """
    print(f"Downloading {repo_id} to {local_dir}...")
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            token=token,
            resume_download=resume,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],  # Ignore non-PyTorch files
        )
        print(f"Successfully downloaded {repo_id}")
        
    except Exception as e:
        print(f"Error downloading {repo_id}: {e}")
        raise


def setup_lora_directories(base_dir: Path) -> None:
    """Create directory structure for LoRAs."""
    domains = [
        "sign_type",
        "mounting",
        "perspective",
        "environment",
        "lighting",
        "material",
    ]
    
    for domain in domains:
        (base_dir / domain).mkdir(parents=True, exist_ok=True)
        # Create a README for each domain
        (base_dir / domain / "README.md").write_text(
            f"# {domain.replace('_', ' ').title()} LoRAs\n\n"
            f"Place `{domain}` adapters here."
        )


def main():
    parser = argparse.ArgumentParser(description="Download SignForge models")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output",
        default="models",
        help="Output directory (default: models)",
    )
    parser.add_argument(
        "--token",
        help="HuggingFace API token (optional)",
    )
    parser.add_argument(
        "--skip-base",
        action="store_true",
        help="Skip base model download",
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output).resolve()
    
    # Setup directories
    base_model_dir = output_dir / "base"
    lora_dir = output_dir / "loras"
    
    base_model_dir.mkdir(parents=True, exist_ok=True)
    lora_dir.mkdir(parents=True, exist_ok=True)
    
    # Download base model
    if not args.skip_base:
        download_model(args.model, base_model_dir, args.token)
    
    # Setup LoRA structure
    setup_lora_directories(lora_dir)
    
    print("\nModel setup complete!")
    print(f"Base model: {base_model_dir}")
    print(f"LoRA directory: {lora_dir}")


if __name__ == "__main__":
    main()
