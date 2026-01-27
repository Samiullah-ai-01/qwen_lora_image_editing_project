"""
Background conditioning for SignForge.

Provides methods for incorporating background images into generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

from PIL import Image, ImageFilter, ImageEnhance

from signforge.core.logging import get_logger

logger = get_logger(__name__)


class BackgroundConditioner:
    """
    Handles background image conditioning for generation.
    
    Supports:
    - ControlNet conditioning (requires compatible model)
    - Image-to-image conditioning
    - Post-generation compositing
    """
    
    def __init__(self, mode: str = "img2img") -> None:
        """
        Initialize the background conditioner.
        
        Args:
            mode: Conditioning mode ('controlnet', 'img2img', 'composite')
        """
        self.mode = mode
    
    def prepare_background(
        self,
        background: Image.Image,
        target_size: tuple[int, int],
        blur: float = 0,
        darken: float = 0,
    ) -> Image.Image:
        """
        Prepare background image for conditioning.
        
        Args:
            background: Input background image
            target_size: Target size (width, height)
            blur: Blur amount (0-10)
            darken: Darken amount (0-1)
            
        Returns:
            Processed background
        """
        # Resize to target
        background = self._resize_cover(background, target_size)
        
        # Apply blur
        if blur > 0:
            background = background.filter(
                ImageFilter.GaussianBlur(radius=blur)
            )
        
        # Apply darkening
        if darken > 0:
            enhancer = ImageEnhance.Brightness(background)
            background = enhancer.enhance(1 - darken)
        
        return background
    
    def _resize_cover(
        self,
        image: Image.Image,
        target_size: tuple[int, int],
    ) -> Image.Image:
        """
        Resize image to cover target size (crop excess).
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            
        Returns:
            Resized and cropped image
        """
        target_w, target_h = target_size
        img_w, img_h = image.size
        
        # Calculate scale to cover
        scale = max(target_w / img_w, target_h / img_h)
        
        # Resize
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        image = image.crop((left, top, left + target_w, top + target_h))
        
        return image
    
    def extract_depth_map(
        self,
        background: Image.Image,
    ) -> Image.Image:
        """
        Extract depth map from background for ControlNet.
        
        Args:
            background: Input background
            
        Returns:
            Grayscale depth map
        """
        # Simple edge-based depth estimation
        # For real depth, use a depth estimation model
        gray = background.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Invert and blur for pseudo-depth
        from PIL import ImageOps
        depth = ImageOps.invert(edges)
        depth = depth.filter(ImageFilter.GaussianBlur(radius=5))
        
        return depth
    
    def extract_canny_edges(
        self,
        background: Image.Image,
        low_threshold: int = 100,
        high_threshold: int = 200,
    ) -> Image.Image:
        """
        Extract Canny edges for ControlNet.
        
        Args:
            background: Input background
            low_threshold: Low edge threshold
            high_threshold: High edge threshold
            
        Returns:
            Edge map
        """
        import numpy as np
        
        # Convert to grayscale
        gray = np.array(background.convert("L"))
        
        try:
            import cv2
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        except ImportError:
            # Fallback without cv2
            gray_img = Image.fromarray(gray)
            edges_img = gray_img.filter(ImageFilter.FIND_EDGES)
            edges = np.array(edges_img)
        
        return Image.fromarray(edges)
    
    def composite_foreground(
        self,
        background: Image.Image,
        foreground: Image.Image,
        mask: Optional[Image.Image] = None,
        blend_mode: str = "normal",
        opacity: float = 1.0,
    ) -> Image.Image:
        """
        Composite foreground onto background.
        
        Args:
            background: Background image
            foreground: Foreground to overlay
            mask: Optional mask for foreground
            blend_mode: Blend mode ('normal', 'multiply', 'screen', 'overlay')
            opacity: Foreground opacity
            
        Returns:
            Composited image
        """
        # Ensure same size
        if foreground.size != background.size:
            foreground = foreground.resize(background.size, Image.Resampling.LANCZOS)
        
        # Convert to RGBA
        background = background.convert("RGBA")
        foreground = foreground.convert("RGBA")
        
        # Apply blend mode
        if blend_mode == "multiply":
            blended = self._blend_multiply(background, foreground)
        elif blend_mode == "screen":
            blended = self._blend_screen(background, foreground)
        elif blend_mode == "overlay":
            blended = self._blend_overlay(background, foreground)
        else:
            blended = foreground
        
        # Apply opacity
        if opacity < 1.0:
            r, g, b, a = blended.split()
            a = a.point(lambda x: int(x * opacity))
            blended = Image.merge("RGBA", (r, g, b, a))
        
        # Apply mask if provided
        if mask is not None:
            if mask.size != background.size:
                mask = mask.resize(background.size, Image.Resampling.LANCZOS)
            mask = mask.convert("L")
            
            # Use mask as alpha
            blended.putalpha(mask)
        
        # Composite
        result = Image.alpha_composite(background, blended)
        
        return result.convert("RGB")
    
    def _blend_multiply(
        self,
        base: Image.Image,
        blend: Image.Image,
    ) -> Image.Image:
        """Multiply blend mode."""
        import numpy as np
        
        base_arr = np.array(base).astype(float) / 255
        blend_arr = np.array(blend).astype(float) / 255
        
        result = base_arr * blend_arr
        result = (result * 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _blend_screen(
        self,
        base: Image.Image,
        blend: Image.Image,
    ) -> Image.Image:
        """Screen blend mode."""
        import numpy as np
        
        base_arr = np.array(base).astype(float) / 255
        blend_arr = np.array(blend).astype(float) / 255
        
        result = 1 - (1 - base_arr) * (1 - blend_arr)
        result = (result * 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def _blend_overlay(
        self,
        base: Image.Image,
        blend: Image.Image,
    ) -> Image.Image:
        """Overlay blend mode."""
        import numpy as np
        
        base_arr = np.array(base).astype(float) / 255
        blend_arr = np.array(blend).astype(float) / 255
        
        # Overlay: multiply where base < 0.5, screen where base >= 0.5
        low_mask = base_arr < 0.5
        result = np.zeros_like(base_arr)
        
        result[low_mask] = 2 * base_arr[low_mask] * blend_arr[low_mask]
        result[~low_mask] = 1 - 2 * (1 - base_arr[~low_mask]) * (1 - blend_arr[~low_mask])
        
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return Image.fromarray(result)
    
    def get_controlnet_conditioning(
        self,
        background: Image.Image,
        control_type: str = "depth",
    ) -> Image.Image:
        """
        Get conditioning image for ControlNet.
        
        Args:
            background: Input background
            control_type: Type of control ('depth', 'canny', 'normal')
            
        Returns:
            Conditioning image
        """
        if control_type == "depth":
            return self.extract_depth_map(background)
        elif control_type == "canny":
            return self.extract_canny_edges(background)
        elif control_type == "normal":
            # For normal maps, would need a proper estimation model
            return self.extract_depth_map(background)
        else:
            raise ValueError(f"Unknown control type: {control_type}")
