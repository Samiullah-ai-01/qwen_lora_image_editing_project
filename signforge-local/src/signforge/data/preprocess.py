"""
Data preprocessing for SignForge.
"""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional
from PIL import Image
from signforge.core.config import get_config
from signforge.core.logging import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """Preprocesses training data."""

    def __init__(
        self,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        resolution: int = 512,
    ) -> None:
        config = get_config()
        self.input_dir = Path(input_dir or config.get_absolute_path(config.data.raw_dir))
        self.output_dir = Path(output_dir or config.get_absolute_path(config.data.processed_dir))
        self.resolution = resolution

    def process_all(self) -> dict:
        """Process all domains and concepts."""
        stats = {"domains": {}, "total_images": 0, "errors": 0}
        
        for domain_dir in self.input_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            
            domain = domain_dir.name
            stats["domains"][domain] = {}
            
            for concept_dir in domain_dir.iterdir():
                if not concept_dir.is_dir():
                    continue
                
                concept = concept_dir.name
                result = self.process_concept(domain, concept)
                stats["domains"][domain][concept] = result
                stats["total_images"] += result["processed"]
                stats["errors"] += result["errors"]
        
        logger.info("preprocessing_complete", **stats)
        return stats

    def process_concept(self, domain: str, concept: str) -> dict:
        """Process a single concept."""
        input_path = self.input_dir / domain / concept
        output_path = self.output_dir / domain / concept
        output_path.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        errors = 0
        image_exts = {".jpg", ".jpeg", ".png", ".webp"}
        
        for img_path in input_path.iterdir():
            if img_path.suffix.lower() not in image_exts:
                continue
            
            try:
                # Process image
                self._process_image(img_path, output_path)
                
                # Copy caption if exists
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    caption = self._normalize_caption(caption_path.read_text(encoding="utf-8"))
                    out_caption = output_path / f"{img_path.stem}.txt"
                    out_caption.write_text(caption, encoding="utf-8")
                
                processed += 1
            except Exception as e:
                logger.warning("process_error", file=str(img_path), error=str(e))
                errors += 1
        
        # Write metadata
        metadata = {
            "domain": domain,
            "concept": concept,
            "processed": processed,
            "resolution": self.resolution,
            "hash": self._compute_hash(output_path),
        }
        (output_path / "metadata.json").write_text(json.dumps(metadata, indent=2))
        
        logger.info("concept_processed", domain=domain, concept=concept, count=processed)
        return {"processed": processed, "errors": errors}

    def _process_image(self, input_path: Path, output_dir: Path) -> None:
        """Process a single image."""
        image = Image.open(input_path).convert("RGB")
        
        # Resize to fit resolution
        w, h = image.size
        scale = self.resolution / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_w - self.resolution) // 2
        top = (new_h - self.resolution) // 2
        image = image.crop((left, top, left + self.resolution, top + self.resolution))
        
        # Save
        output_path = output_dir / f"{input_path.stem}.png"
        image.save(output_path, "PNG")

    def _normalize_caption(self, caption: str) -> str:
        """Normalize caption text."""
        caption = caption.strip()
        caption = " ".join(caption.split())  # Normalize whitespace
        return caption

    def _compute_hash(self, dir_path: Path) -> str:
        """Compute hash of dataset for versioning."""
        hasher = hashlib.sha256()
        for f in sorted(dir_path.glob("*.png")):
            hasher.update(f.read_bytes())
        return hasher.hexdigest()[:16]


def main():
    """CLI entry point for preprocessing."""
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess training data")
    parser.add_argument("--input", type=Path, help="Input directory")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(args.input, args.output, args.resolution)
    preprocessor.process_all()


if __name__ == "__main__":
    main()
