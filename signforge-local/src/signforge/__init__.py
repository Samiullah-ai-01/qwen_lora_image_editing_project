"""
SignForge Local - Signage Mockup Generator with Multi-LoRA Composition

A local-first ML pipeline for generating professional signage mockups using
modular LoRA adapters for sign types, mounting styles, perspectives, and environments.
"""

__version__ = "0.1.0"
__author__ = "SignForge Contributors"

from signforge.core.config import get_config
from signforge.core.logging import get_logger

__all__ = ["__version__", "get_config", "get_logger"]
