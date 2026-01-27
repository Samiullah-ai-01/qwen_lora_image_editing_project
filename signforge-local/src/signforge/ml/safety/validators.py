"""
Input validation and safety checks for SignForge.

Validates prompts, images, and requests for safety and correctness.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from PIL import Image

from signforge.core.config import get_config
from signforge.core.errors import ValidationError, SafetyError
from signforge.core.logging import get_logger

logger = get_logger(__name__)


class PromptValidator:
    """Validates and sanitizes prompts."""
    
    def __init__(self) -> None:
        """Initialize the validator."""
        config = get_config()
        self.max_length = config.safety.max_prompt_length
        self.blocked_words = set(w.lower() for w in config.safety.blocked_words)
    
    def validate(self, prompt: str) -> str:
        """
        Validate and sanitize a prompt.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Sanitized prompt
            
        Raises:
            ValidationError: If prompt is invalid
            SafetyError: If prompt contains blocked content
        """
        if not prompt:
            raise ValidationError("Prompt cannot be empty", field="prompt")
        
        # Strip and normalize whitespace
        prompt = " ".join(prompt.split())
        
        # Check length
        if len(prompt) > self.max_length:
            raise ValidationError(
                f"Prompt too long ({len(prompt)} > {self.max_length})",
                field="prompt",
                value=prompt[:100] + "...",
            )
        
        # Check for blocked words
        prompt_lower = prompt.lower()
        for word in self.blocked_words:
            if word in prompt_lower:
                logger.warning("blocked_word_detected", word=word)
                raise SafetyError(
                    "Prompt contains blocked content",
                    reason="blocked_word",
                )
        
        # Remove potentially dangerous characters
        prompt = self._sanitize(prompt)
        
        return prompt
    
    def _sanitize(self, text: str) -> str:
        """Remove potentially dangerous characters."""
        # Remove control characters except newline/tab
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        
        # Limit consecutive special characters
        text = re.sub(r'([^\w\s])\1{3,}', r'\1\1\1', text)
        
        return text


class ImageValidator:
    """Validates uploaded images."""
    
    def __init__(self) -> None:
        """Initialize the validator."""
        config = get_config()
        self.allowed_formats = set(config.safety.allowed_image_formats)
        self.max_width = config.safety.max_upload_width
        self.max_height = config.safety.max_upload_height
    
    def validate(
        self,
        image: Image.Image,
        source: str = "upload",
    ) -> Image.Image:
        """
        Validate an image.
        
        Args:
            image: PIL Image to validate
            source: Source description for logging
            
        Returns:
            Validated image (possibly converted)
            
        Raises:
            ValidationError: If image is invalid
        """
        # Check dimensions
        if image.width > self.max_width:
            raise ValidationError(
                f"Image too wide ({image.width} > {self.max_width})",
                field="image",
            )
        
        if image.height > self.max_height:
            raise ValidationError(
                f"Image too tall ({image.height} > {self.max_height})",
                field="image",
            )
        
        # Ensure valid mode
        if image.mode not in ("RGB", "RGBA", "L"):
            image = image.convert("RGB")
        
        logger.debug(
            "image_validated",
            source=source,
            size=f"{image.width}x{image.height}",
            mode=image.mode,
        )
        
        return image
    
    def validate_file(self, path: Path) -> Image.Image:
        """
        Validate an image file.
        
        Args:
            path: Path to image file
            
        Returns:
            Loaded and validated image
            
        Raises:
            ValidationError: If file is invalid
        """
        path = Path(path)
        
        # Check path traversal
        try:
            path = path.resolve()
        except Exception:
            raise ValidationError("Invalid file path", field="path")
        
        # Check exists
        if not path.exists():
            raise ValidationError("File not found", field="path", value=str(path))
        
        # Check extension
        ext = path.suffix.lower().lstrip(".")
        if ext not in self.allowed_formats:
            raise ValidationError(
                f"Invalid image format: {ext}",
                field="format",
                value=ext,
            )
        
        # Try to load
        try:
            image = Image.open(path)
            image.load()  # Force load to catch corrupt images
        except Exception as e:
            raise ValidationError(f"Failed to load image: {e}", field="image")
        
        return self.validate(image, source=str(path))


class RequestValidator:
    """Validates generation requests."""
    
    def __init__(self) -> None:
        """Initialize the validator."""
        config = get_config()
        self.max_resolution = config.inference.max_resolution
        self.min_dimension = 256
        self.max_dimension = 2048
        self.max_steps = 100
        self.max_adapters = 6
        
        self.prompt_validator = PromptValidator()
        self.image_validator = ImageValidator()
    
    def validate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 768,
        steps: int = 30,
        guidance_scale: float = 7.5,
        adapters: Optional[list[str]] = None,
        adapter_weights: Optional[list[float]] = None,
        logo_image: Optional[Image.Image] = None,
        background_image: Optional[Image.Image] = None,
    ) -> dict:
        """
        Validate a complete generation request.
        
        Args:
            prompt: Generation prompt
            negative_prompt: Negative prompt
            width: Image width
            height: Image height
            steps: Number of steps
            guidance_scale: Guidance scale
            adapters: Adapter names
            adapter_weights: Adapter weights
            logo_image: Optional logo image
            background_image: Optional background image
            
        Returns:
            Validated parameters dict
            
        Raises:
            ValidationError: If any parameter is invalid
        """
        # Validate prompt
        prompt = self.prompt_validator.validate(prompt)
        
        if negative_prompt:
            negative_prompt = self.prompt_validator.validate(negative_prompt)
        
        # Validate dimensions
        self._validate_dimensions(width, height)
        
        # Validate steps
        if steps < 1:
            raise ValidationError("Steps must be at least 1", field="steps", value=steps)
        if steps > self.max_steps:
            raise ValidationError(
                f"Steps too high ({steps} > {self.max_steps})",
                field="steps",
                value=steps,
            )
        
        # Validate guidance
        if guidance_scale < 1 or guidance_scale > 30:
            raise ValidationError(
                "Guidance scale must be between 1 and 30",
                field="guidance_scale",
                value=guidance_scale,
            )
        
        # Validate adapters
        adapters = adapters or []
        adapter_weights = adapter_weights or []
        
        if len(adapters) > self.max_adapters:
            raise ValidationError(
                f"Too many adapters ({len(adapters)} > {self.max_adapters})",
                field="adapters",
            )
        
        if len(adapter_weights) != len(adapters):
            # Pad with defaults
            while len(adapter_weights) < len(adapters):
                adapter_weights.append(1.0)
            adapter_weights = adapter_weights[:len(adapters)]
        
        # Validate adapter weights
        for i, weight in enumerate(adapter_weights):
            if weight < 0 or weight > 2:
                raise ValidationError(
                    "Adapter weight must be between 0 and 2",
                    field=f"adapter_weights[{i}]",
                    value=weight,
                )
        
        # Validate images
        if logo_image is not None:
            logo_image = self.image_validator.validate(logo_image, source="logo")
        
        if background_image is not None:
            background_image = self.image_validator.validate(
                background_image, source="background"
            )
        
        return {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "adapters": adapters,
            "adapter_weights": adapter_weights,
            "logo_image": logo_image,
            "background_image": background_image,
        }
    
    def _validate_dimensions(self, width: int, height: int) -> None:
        """Validate image dimensions."""
        if width < self.min_dimension:
            raise ValidationError(
                f"Width too small ({width} < {self.min_dimension})",
                field="width",
                value=width,
            )
        
        if width > self.max_dimension:
            raise ValidationError(
                f"Width too large ({width} > {self.max_dimension})",
                field="width",
                value=width,
            )
        
        if height < self.min_dimension:
            raise ValidationError(
                f"Height too small ({height} < {self.min_dimension})",
                field="height",
                value=height,
            )
        
        if height > self.max_dimension:
            raise ValidationError(
                f"Height too large ({height} > {self.max_dimension})",
                field="height",
                value=height,
            )
        
        if width * height > self.max_resolution:
            raise ValidationError(
                f"Resolution too high ({width}x{height} = {width*height} > {self.max_resolution})",
                field="resolution",
            )
        
        # Ensure dimensions are multiples of 8 (for VAE)
        if width % 8 != 0:
            raise ValidationError(
                "Width must be a multiple of 8",
                field="width",
                value=width,
            )
        
        if height % 8 != 0:
            raise ValidationError(
                "Height must be a multiple of 8",
                field="height",
                value=height,
            )


# Convenience functions
_prompt_validator: Optional[PromptValidator] = None
_image_validator: Optional[ImageValidator] = None


def validate_prompt(prompt: str) -> str:
    """Validate a prompt using the default validator."""
    global _prompt_validator
    if _prompt_validator is None:
        _prompt_validator = PromptValidator()
    return _prompt_validator.validate(prompt)


def validate_image(image: Image.Image) -> Image.Image:
    """Validate an image using the default validator."""
    global _image_validator
    if _image_validator is None:
        _image_validator = ImageValidator()
    return _image_validator.validate(image)
