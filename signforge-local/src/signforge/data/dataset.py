"""
Dataset definitions for SignForge training.
"""

from __future__ import annotations
import random
from pathlib import Path
from typing import Optional, Callable
from PIL import Image
import torch
from torch.utils.data import Dataset
from signforge.data.schema import DatasetItem
from signforge.core.logging import get_logger

logger = get_logger(__name__)


class SignForgeDataset(Dataset):
    """Dataset for LoRA training."""

    def __init__(
        self,
        data_dir: Path,
        resolution: int = 512,
        center_crop: bool = True,
        random_flip: bool = True,
        caption_dropout: float = 0.1,
        transform: Optional[Callable] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.caption_dropout = caption_dropout
        self.transform = transform
        
        self.items: list[DatasetItem] = []
        self._scan_directory()

    def _scan_directory(self) -> None:
        """Scan directory for image-caption pairs."""
        image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
        
        for img_path in self.data_dir.rglob("*"):
            if img_path.suffix.lower() not in image_extensions:
                continue
            
            # Look for caption file
            caption_path = img_path.with_suffix(".txt")
            caption = ""
            if caption_path.exists():
                caption = caption_path.read_text(encoding="utf-8").strip()
            
            self.items.append(DatasetItem(
                image_path=img_path,
                caption=caption,
            ))
        
        logger.info("dataset_scanned", count=len(self.items), path=str(self.data_dir))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        item = self.items[idx]
        
        # Load image
        image = Image.open(item.image_path).convert("RGB")
        
        # Resize and crop
        image = self._resize_crop(image)
        
        # Random flip
        if self.random_flip and random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Caption dropout
        caption = item.caption
        if self.caption_dropout > 0 and random.random() < self.caption_dropout:
            caption = ""
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return {
            "image": image,
            "caption": caption,
            "path": str(item.image_path),
        }

    def _resize_crop(self, image: Image.Image) -> Image.Image:
        """Resize and optionally center crop."""
        w, h = image.size
        
        # Scale to fit resolution
        scale = self.resolution / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop
        if self.center_crop:
            left = (new_w - self.resolution) // 2
            top = (new_h - self.resolution) // 2
            image = image.crop((left, top, left + self.resolution, top + self.resolution))
        
        return image

    def get_all_captions(self) -> list[str]:
        """Get all captions for vocabulary building."""
        return [item.caption for item in self.items if item.caption]

    def split(self, val_ratio: float = 0.1) -> tuple["SignForgeDataset", "SignForgeDataset"]:
        """Split into train and validation sets."""
        n_val = int(len(self.items) * val_ratio)
        indices = list(range(len(self.items)))
        random.shuffle(indices)
        
        val_items = [self.items[i] for i in indices[:n_val]]
        train_items = [self.items[i] for i in indices[n_val:]]
        
        train_ds = SignForgeDataset.__new__(SignForgeDataset)
        train_ds.__dict__.update(self.__dict__)
        train_ds.items = train_items
        
        val_ds = SignForgeDataset.__new__(SignForgeDataset)
        val_ds.__dict__.update(self.__dict__)
        val_ds.items = val_items
        
        return train_ds, val_ds
