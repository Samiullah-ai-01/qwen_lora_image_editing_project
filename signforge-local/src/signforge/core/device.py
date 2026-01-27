"""
Device management for SignForge.

Handles GPU detection, memory management, and device selection.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

import torch

from signforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GPUInfo:
    """Information about an available GPU."""
    
    index: int
    name: str
    total_memory: int  # bytes
    allocated_memory: int  # bytes
    cached_memory: int  # bytes
    free_memory: int  # bytes
    compute_capability: tuple[int, int]
    supports_bf16: bool


class DeviceManager:
    """
    Manages device selection and GPU resources.
    
    Provides utilities for:
    - GPU detection and selection
    - Memory monitoring
    - Device-appropriate dtype selection
    - Memory cleanup
    """
    
    _instance: Optional["DeviceManager"] = None
    
    def __new__(cls) -> "DeviceManager":
        """Singleton pattern for device manager."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the device manager."""
        if self._initialized:
            return
            
        self._initialized = True
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None
        self._gpu_info: Optional[GPUInfo] = None
        
        self._detect_device()
    
    def _detect_device(self) -> None:
        """Detect available compute devices."""
        if torch.cuda.is_available():
            device_index = 0
            self._device = torch.device(f"cuda:{device_index}")
            
            # Get GPU information
            props = torch.cuda.get_device_properties(device_index)
            total_memory = props.total_memory
            allocated = torch.cuda.memory_allocated(device_index)
            cached = torch.cuda.memory_reserved(device_index)
            
            self._gpu_info = GPUInfo(
                index=device_index,
                name=props.name,
                total_memory=total_memory,
                allocated_memory=allocated,
                cached_memory=cached,
                free_memory=total_memory - allocated,
                compute_capability=(props.major, props.minor),
                supports_bf16=props.major >= 8,  # Ampere+
            )
            
            # Select appropriate dtype
            if self._gpu_info.supports_bf16:
                self._dtype = torch.bfloat16
            else:
                self._dtype = torch.float16
                
            logger.info(
                "gpu_detected",
                name=self._gpu_info.name,
                memory_gb=round(total_memory / (1024**3), 2),
                dtype=str(self._dtype),
                bf16_support=self._gpu_info.supports_bf16,
            )
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            logger.warning("no_gpu_detected", message="Running on CPU - expect slow performance")
    
    @property
    def device(self) -> torch.device:
        """Get the current compute device."""
        return self._device
    
    @property
    def dtype(self) -> torch.dtype:
        """Get the recommended dtype for the device."""
        return self._dtype
    
    @property
    def gpu_info(self) -> Optional[GPUInfo]:
        """Get GPU information if available."""
        return self._gpu_info
    
    @property
    def is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        return torch.cuda.is_available()
    
    @property
    def is_bf16_supported(self) -> bool:
        """Check if bfloat16 is supported."""
        if self._gpu_info:
            return self._gpu_info.supports_bf16
        return False
    
    def get_memory_info(self) -> dict:
        """Get current GPU memory information."""
        if not self.is_cuda_available:
            return {
                "available": False,
                "device": "cpu",
            }
        
        device_index = 0 if self._device.index is None else self._device.index
        
        allocated = torch.cuda.memory_allocated(device_index)
        cached = torch.cuda.memory_reserved(device_index)
        total = torch.cuda.get_device_properties(device_index).total_memory
        
        return {
            "available": True,
            "device": self._gpu_info.name if self._gpu_info else "unknown",
            "total_bytes": total,
            "allocated_bytes": allocated,
            "cached_bytes": cached,
            "free_bytes": total - allocated,
            "total_gb": round(total / (1024**3), 2),
            "allocated_gb": round(allocated / (1024**3), 2),
            "cached_gb": round(cached / (1024**3), 2),
            "free_gb": round((total - allocated) / (1024**3), 2),
            "utilization_percent": round(allocated / total * 100, 1),
        }
    
    def clear_cache(self) -> None:
        """Clear GPU cache and run garbage collection."""
        gc.collect()
        if self.is_cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("cache_cleared")
    
    def set_device(self, device: str) -> None:
        """
        Explicitly set the compute device.
        
        Args:
            device: Device string (e.g., 'cuda', 'cuda:0', 'cpu', 'auto')
        """
        if device == "auto":
            self._detect_device()
        elif device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            self._device = torch.device(device)
            
            # Update GPU info for new device
            device_index = 0 if ":" not in device else int(device.split(":")[1])
            props = torch.cuda.get_device_properties(device_index)
            
            self._gpu_info = GPUInfo(
                index=device_index,
                name=props.name,
                total_memory=props.total_memory,
                allocated_memory=torch.cuda.memory_allocated(device_index),
                cached_memory=torch.cuda.memory_reserved(device_index),
                free_memory=props.total_memory - torch.cuda.memory_allocated(device_index),
                compute_capability=(props.major, props.minor),
                supports_bf16=props.major >= 8,
            )
            
            if self._gpu_info.supports_bf16:
                self._dtype = torch.bfloat16
            else:
                self._dtype = torch.float16
        else:
            self._device = torch.device("cpu")
            self._dtype = torch.float32
            self._gpu_info = None
        
        logger.info("device_set", device=str(self._device), dtype=str(self._dtype))
    
    def set_dtype(self, dtype: str) -> None:
        """
        Explicitly set the dtype.
        
        Args:
            dtype: Dtype string ('bf16', 'fp16', 'fp32')
        """
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        
        if dtype not in dtype_map:
            raise ValueError(f"Unknown dtype: {dtype}. Use one of {list(dtype_map.keys())}")
        
        if dtype == "bf16" and not self.is_bf16_supported:
            logger.warning("bf16_not_supported", message="GPU does not support bf16, using fp16")
            self._dtype = torch.float16
        else:
            self._dtype = dtype_map[dtype]
        
        logger.info("dtype_set", dtype=str(self._dtype))
    
    def get_recommended_settings(self) -> dict:
        """Get recommended settings based on GPU capabilities."""
        if not self.is_cuda_available:
            return {
                "use_xformers": False,
                "attention_slicing": True,
                "vae_tiling": True,
                "gradient_checkpointing": True,
                "batch_size": 1,
            }
        
        memory = self.get_memory_info()
        total_gb = memory["total_gb"]
        
        if total_gb >= 24:
            # High-end GPU (RTX 4090, A100)
            return {
                "use_xformers": True,
                "attention_slicing": False,
                "vae_tiling": False,
                "gradient_checkpointing": False,
                "batch_size": 4,
            }
        elif total_gb >= 16:
            # Mid-range (RTX 4080, 3090)
            return {
                "use_xformers": True,
                "attention_slicing": False,
                "vae_tiling": False,
                "gradient_checkpointing": True,
                "batch_size": 2,
            }
        elif total_gb >= 12:
            # Entry (RTX 4070, 3080)
            return {
                "use_xformers": True,
                "attention_slicing": True,
                "vae_tiling": False,
                "gradient_checkpointing": True,
                "batch_size": 1,
            }
        else:
            # Low VRAM
            return {
                "use_xformers": True,
                "attention_slicing": True,
                "vae_tiling": True,
                "gradient_checkpointing": True,
                "batch_size": 1,
            }


def get_device_manager() -> DeviceManager:
    """Get the singleton device manager instance."""
    return DeviceManager()
