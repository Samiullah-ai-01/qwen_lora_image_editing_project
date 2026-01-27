"""
Logo conditioning for SignForge.

Provides methods for incorporating logo images into generation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Any

import numpy as np
from PIL import Image

from signforge.core.logging import get_logger

logger = get_logger(__name__)


class LogoConditioner:
    """
    Handles logo image conditioning for generation.
    
    Supports two modes:
    - IP-Adapter style conditioning (requires compatible model)
    - Compositing-based overlay (works with any model)
    """
    
    def __init__(self, mode: str = "composite") -> None:
        """
        Initialize the logo conditioner.
        
        Args:
            mode: Conditioning mode ('ip_adapter' or 'composite')
        """
        self.mode = mode
        self._ip_adapter_loaded = False
    
    def prepare_logo(
        self,
        logo: Image.Image,
        target_size: Optional[tuple[int, int]] = None,
        padding: int = 20,
    ) -> Image.Image:
        """
        Prepare a logo image for use.
        
        Args:
            logo: Input logo image
            target_size: Optional target size (width, height)
            padding: Padding to add around logo
            
        Returns:
            Processed logo image
        """
        # Convert to RGBA if needed
        if logo.mode != "RGBA":
            logo = logo.convert("RGBA")
        
        # Resize if target size specified
        if target_size:
            # Calculate size maintaining aspect ratio
            logo_aspect = logo.width / logo.height
            target_w, target_h = target_size
            target_w -= padding * 2
            target_h -= padding * 2
            
            if logo_aspect > target_w / target_h:
                new_w = target_w
                new_h = int(target_w / logo_aspect)
            else:
                new_h = target_h
                new_w = int(target_h * logo_aspect)
            
            logo = logo.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        return logo
    
    def create_logo_mask(
        self,
        logo: Image.Image,
        feather: int = 5,
    ) -> Image.Image:
        """
        Create a mask from logo alpha channel.
        
        Args:
            logo: Logo image with alpha channel
            feather: Feather amount for soft edges
            
        Returns:
            Grayscale mask image
        """
        if logo.mode != "RGBA":
            # Return full mask if no alpha
            return Image.new("L", logo.size, 255)
        
        # Extract alpha channel
        alpha = logo.split()[3]
        
        # Apply feathering if requested
        if feather > 0:
            from PIL import ImageFilter
            alpha = alpha.filter(ImageFilter.GaussianBlur(feather))
        
        return alpha
    
    def composite_logo(
        self,
        base_image: Image.Image,
        logo: Image.Image,
        position: str = "center",
        scale: float = 0.3,
        opacity: float = 1.0,
    ) -> Image.Image:
        """
        Composite logo onto base image.
        
        Args:
            base_image: Background image
            logo: Logo to overlay
            position: Position ('center', 'top', 'bottom', etc.)
            scale: Logo scale relative to image width
            opacity: Logo opacity (0-1)
            
        Returns:
            Composited image
        """
        # Convert base to RGBA
        if base_image.mode != "RGBA":
            base_image = base_image.convert("RGBA")
        
        # Prepare logo
        logo = self.prepare_logo(logo)
        
        # Calculate logo size
        target_w = int(base_image.width * scale)
        logo_aspect = logo.width / logo.height
        target_h = int(target_w / logo_aspect)
        
        logo = logo.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        # Apply opacity
        if opacity < 1.0:
            r, g, b, a = logo.split()
            a = a.point(lambda x: int(x * opacity))
            logo = Image.merge("RGBA", (r, g, b, a))
        
        # Calculate position
        x, y = self._calculate_position(
            base_image.size,
            logo.size,
            position,
        )
        
        # Create composite
        result = base_image.copy()
        result.paste(logo, (x, y), logo)
        
        return result.convert("RGB")
    
    def _calculate_position(
        self,
        base_size: tuple[int, int],
        overlay_size: tuple[int, int],
        position: str,
    ) -> tuple[int, int]:
        """Calculate overlay position."""
        bw, bh = base_size
        ow, oh = overlay_size
        margin = 20
        
        positions = {
            "center": ((bw - ow) // 2, (bh - oh) // 2),
            "top": ((bw - ow) // 2, margin),
            "bottom": ((bw - ow) // 2, bh - oh - margin),
            "left": (margin, (bh - oh) // 2),
            "right": (bw - ow - margin, (bh - oh) // 2),
            "top_left": (margin, margin),
            "top_right": (bw - ow - margin, margin),
            "bottom_left": (margin, bh - oh - margin),
            "bottom_right": (bw - ow - margin, bh - oh - margin),
        }
        
        return positions.get(position, positions["center"])
    
    def apply_perspective_warp(
        self,
        logo: Image.Image,
        corners: list[tuple[int, int]],
    ) -> Image.Image:
        """
        Apply perspective warp to logo.
        
        Args:
            logo: Logo image
            corners: Four corner points [top_left, top_right, bottom_right, bottom_left]
            
        Returns:
            Warped logo image
        """
        if len(corners) != 4:
            raise ValueError("Must provide exactly 4 corner points")
        
        # Calculate bounding box
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        
        width = max_x - min_x
        height = max_y - min_y
        
        # Source corners (logo corners)
        src_corners = [
            (0, 0),
            (logo.width, 0),
            (logo.width, logo.height),
            (0, logo.height),
        ]
        
        # Adjust target corners relative to bounding box
        dst_corners = [(x - min_x, y - min_y) for x, y in corners]
        
        # Calculate perspective transform coefficients
        coeffs = self._find_perspective_coeffs(dst_corners, src_corners)
        
        # Apply transform
        warped = logo.transform(
            (width, height),
            Image.Transform.PERSPECTIVE,
            coeffs,
            Image.Resampling.BICUBIC,
        )
        
        return warped
    
    def _find_perspective_coeffs(
        self,
        target_coords: list[tuple[int, int]],
        source_coords: list[tuple[int, int]],
    ) -> tuple:
        """Calculate perspective transform coefficients."""
        matrix = []
        for s, t in zip(source_coords, target_coords):
            matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0]*t[0], -s[0]*t[1]])
            matrix.append([0, 0, 0, t[0], t[1], 1, -s[1]*t[0], -s[1]*t[1]])
        
        A = np.matrix(matrix, dtype=float)
        B = np.array([c for p in source_coords for c in p]).reshape(8)
        
        res = np.linalg.solve(A, B)
        return tuple(res.flat)
    
    def get_ip_adapter_embeds(
        self,
        logo: Image.Image,
        pipeline: Any,
    ) -> Any:
        """
        Get IP-Adapter embeddings for logo.
        
        Args:
            logo: Logo image
            pipeline: Diffusion pipeline with IP-Adapter
            
        Returns:
            Image embeddings for conditioning
        """
        # This requires IP-Adapter to be loaded in the pipeline
        # Implement based on specific IP-Adapter version
        raise NotImplementedError(
            "IP-Adapter conditioning not yet implemented. Use composite mode."
        )
