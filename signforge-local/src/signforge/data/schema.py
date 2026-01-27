"""
Data schema definitions.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class DatasetItem:
    """Single item in the dataset."""
    image_path: Path
    caption: str = ""
    concept: Optional[str] = None
    domain: Optional[str] = None


@dataclass
class SplitInfo:
    """Information about dataset splits."""
    train_count: int
    val_count: int
    test_count: int = 0
    split_seed: int = 42
