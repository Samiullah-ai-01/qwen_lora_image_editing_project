"""
Caption management and processing for SignForge datasets.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List

from signforge.core.logging import get_logger

logger = get_logger(__name__)


class CaptionProcessor:
    """Processes and cleans sign-specific captions."""
    
    def __init__(self, use_tags: bool = True) -> None:
        self.use_tags = use_tags
        # Common artifacts to remove
        self.noise_patterns = [
            r"https?://\S+",
            r"@\w+",
            r"#\w+",
            r"\d+p#", # DPI tags
        ]

    def clean(self, caption: str) -> str:
        """Clean a raw caption."""
        cleaned = caption.strip()
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        
        # Normalize whitespace
        cleaned = " ".join(cleaned.split())
        return cleaned

    def tag_to_natural(self, tags: str) -> str:
        """Convert comma-separated tags to a natural language sentence."""
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]
        if not tag_list:
            return ""
        
        # Simple heuristic: join with commas and a final 'and'
        if len(tag_list) == 1:
            return f"A photograph of a {tag_list[0]}."
        
        main_subject = tag_list[0]
        descriptors = tag_list[1:]
        
        return f"A photograph of a {main_subject} with {', '.join(descriptors[:-1])} and {descriptors[-1]}."

    def extract_text_content(self, caption: str) -> Optional[str]:
        """Attempt to extract the literal text mention from a caption."""
        match = re.search(r"['\"]([^'\"]+)['\"]", caption)
        if match:
            return match.group(1)
        return None


def read_caption(path: Path) -> str:
    """Read a caption from a file."""
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.error("caption_read_error", path=str(path), error=str(e))
        return ""


def write_caption(path: Path, caption: str) -> None:
    """Write a caption to a file."""
    try:
        path.write_text(caption, encoding="utf-8")
    except Exception as e:
        logger.error("caption_write_error", path=str(path), error=str(e))
