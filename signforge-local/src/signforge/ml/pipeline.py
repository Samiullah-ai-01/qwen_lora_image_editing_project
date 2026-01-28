"""
Main image generation pipeline for SignForge.

Handles model loading, LoRA composition, and image generation.
"""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Callable

import torch
from PIL import Image

from signforge.core.config import get_config, AppConfig
from signforge.core.device import DeviceManager, get_device_manager
from signforge.core.logging import get_logger
from signforge.core.errors import ModelError, InferenceError, OutOfMemoryError

logger = get_logger(__name__)


@dataclass
class GenerationRequest:
    """Request for image generation."""
    
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 768
    steps: int = 30
    guidance_scale: float = 7.5
    seed: int = -1
    
    # LoRA adapters
    adapters: list[str] = field(default_factory=list)
    adapter_weights: list[float] = field(default_factory=list)
    normalize_weights: bool = True
    
    # Conditioning images
    logo_image: Optional[Image.Image] = None
    logo_strength: float = 0.8
    background_image: Optional[Image.Image] = None
    background_strength: float = 0.6
    
    # Profile override
    profile: Optional[str] = None
    
    def validate(self) -> None:
        """Validate the request parameters."""
        if not self.prompt or not self.prompt.strip():
            raise InferenceError("Prompt cannot be empty", step="validation")
        
        if len(self.prompt) > 1000:
            raise InferenceError("Prompt too long (max 1000 chars)", step="validation")
        
        if self.width < 256 or self.width > 2048:
            raise InferenceError(f"Invalid width: {self.width} (256-2048)", step="validation")
        
        if self.height < 256 or self.height > 2048:
            raise InferenceError(f"Invalid height: {self.height} (256-2048)", step="validation")
        
        if self.width * self.height > 2097152:
            raise InferenceError("Resolution too high (max 2MP)", step="validation")
        
        if self.steps < 1 or self.steps > 100:
            raise InferenceError(f"Invalid steps: {self.steps} (1-100)", step="validation")
        
        if self.guidance_scale < 1 or self.guidance_scale > 20:
            raise InferenceError(f"Invalid guidance: {self.guidance_scale} (1-20)", step="validation")
        
        if len(self.adapters) != len(self.adapter_weights):
            raise InferenceError(
                "Adapter and weight count mismatch",
                step="validation",
                details={"adapters": len(self.adapters), "weights": len(self.adapter_weights)},
            )


@dataclass
class GenerationResult:
    """Result of image generation."""
    
    image: Image.Image
    seed: int
    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance_scale: float
    adapters: list[str]
    adapter_weights: list[float]
    generation_time_ms: int
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "seed": self.seed,
            "prompt": self.prompt,
            "negative_prompt": self.negative_prompt,
            "width": self.width,
            "height": self.height,
            "steps": self.steps,
            "guidance_scale": self.guidance_scale,
            "adapters": self.adapters,
            "adapter_weights": self.adapter_weights,
            "generation_time_ms": self.generation_time_ms,
        }


class SignForgePipeline:
    """
    Main pipeline for SignForge image generation.
    
    Wraps diffusers pipeline with:
    - Automatic device/dtype selection
    - Memory optimizations
    - LoRA composition
    - Progress callbacks
    """
    
    _instance: Optional["SignForgePipeline"] = None
    
    def __new__(cls) -> "SignForgePipeline":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the pipeline."""
        if self._initialized:
            return
        
        self._initialized = True
        self._config = get_config()
        self._device_manager = get_device_manager()
        self._pipe: Any = None
        self._loaded_adapters: list[str] = []
        self._loading = False
        
        # Don't auto-load - wait for explicit load call
        logger.info("pipeline_created", device=str(self._device_manager.device))
    
    @property
    def is_loaded(self) -> bool:
        """Check if the pipeline is loaded."""
        return self._pipe is not None
    
    @property
    def is_mock(self) -> bool:
        """Check if using mock pipeline."""
        return False
    
    def load(self) -> None:
        """
        Load the base model.
        """
        if self._pipe is not None:
            logger.debug("pipeline_already_loaded")
            return
        
        if self._loading:
            logger.debug("pipeline_load_in_progress")
            return

        self._loading = True
        
        from signforge.version import __version__
        model_path = self._config.get_absolute_path(self._config.model.base_path)
        logger.info("pipeline_loading_start", version=__version__, path=str(model_path))
        
        # Verify model files exist
        if not self.is_mock and not (model_path / "model_index.json").exists():
            logger.error("model_files_missing", path=str(model_path))
            print("\n" + "!"*60)
            print("  FATAL ERROR: BASE MODEL FILES NOT FOUND")
            print(f"  Expected at: {model_path}")
            print("\n  Please run: python scripts/download_models.py")
            print("!"*60 + "\n")
            raise FileNotFoundError(f"Model files not found at {model_path}")
        
        try:
            from diffusers import StableDiffusionXLPipeline
            
            # Load the pipeline
            load_kwargs = {
                "torch_dtype": self._device_manager.dtype,
                "use_safetensors": True,
            }
            
            # If on CPU, variant="fp16" should be avoided if using fp32
            if self._device_manager.is_cuda_available and self._device_manager.dtype == torch.float16:
                load_kwargs["variant"] = "fp16"

            if model_path.exists():
                logger.info("loading_from_local_disk", path=str(model_path))
                self._pipe = StableDiffusionXLPipeline.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                    **load_kwargs
                )
            else:
                logger.info("downloading_from_hub", model_id=self._config.model.hf_model_id)
                self._pipe = StableDiffusionXLPipeline.from_pretrained(
                    self._config.model.hf_model_id,
                    **load_kwargs
                )
                logger.info("saving_model_to_local_disk", path=str(model_path))
                self._pipe.save_pretrained(str(model_path))
            
            # Move to device
            self._pipe = self._pipe.to(self._device_manager.device)
            
            # Apply optimizations
            self._apply_optimizations()
            
            logger.info(
                "model_loaded",
                device=str(self._device_manager.device),
                dtype=str(self._device_manager.dtype),
            )
            
        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            raise ModelError(f"Failed to load diffusion model: {e}")
        finally:
            self._loading = False
    
    @property
    def is_loading(self) -> bool:
        """Check if the pipeline is currently loading."""
        return self._loading
    
    def _apply_optimizations(self) -> None:
        """Apply memory and performance optimizations."""
        settings = self._device_manager.get_recommended_settings()
        
        # xformers
        if settings["use_xformers"] and self._config.model.use_xformers:
            try:
                self._pipe.enable_xformers_memory_efficient_attention()
                logger.debug("xformers_enabled")
            except Exception as e:
                logger.warning("xformers_failed", error=str(e))
        
        # Attention slicing
        if settings["attention_slicing"]:
            slice_size = self._config.model.attention_slicing
            if slice_size == "auto":
                self._pipe.enable_attention_slicing()
            elif slice_size == "max":
                self._pipe.enable_attention_slicing("max")
            elif isinstance(slice_size, int):
                self._pipe.enable_attention_slicing(slice_size)
            logger.debug("attention_slicing_enabled", slice_size=slice_size)
        
        # VAE tiling & slicing (Critical for low VRAM high-res)
        if settings["vae_tiling"] or self._config.model.vae_tiling:
            self._pipe.enable_vae_tiling()
            self._pipe.enable_vae_slicing()
            logger.debug("vae_tiling_and_slicing_enabled")
        
        # Model CPU Offload (For extremely low VRAM systems)
        if self._config.model.cpu_offload and self._device_manager.is_cuda_available:
            try:
                # This moves parts of the model to CPU when not in use
                self._pipe.enable_model_cpu_offload()
                logger.info("sequential_cpu_offload_enabled")
            except Exception as e:
                logger.warning("cpu_offload_failed", error=str(e))
    
    def unload(self) -> None:
        """Unload the model and free memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            self._loaded_adapters.clear()
            self._device_manager.clear_cache()
            logger.info("model_unloaded")
    
    def set_adapters(
        self,
        adapter_names: list[str],
        adapter_weights: list[float],
        normalize: bool = True,
    ) -> None:
        """
        Set active LoRA adapters.
        
        Args:
            adapter_names: List of adapter names to enable
            adapter_weights: Corresponding weights
            normalize: Whether to normalize weights
        """
        if not adapter_names:
            # Disable all adapters
            if hasattr(self._pipe, "unload_lora_weights"):
                self._pipe.unload_lora_weights()
            self._loaded_adapters.clear()
            return
        
        # Normalize weights if requested
        if normalize and adapter_weights:
            total = sum(adapter_weights)
            if total > 0:
                adapter_weights = [w / total for w in adapter_weights]
        
        try:
            self._pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
            self._loaded_adapters = adapter_names
            
            logger.debug(
                "adapters_set",
                adapters=adapter_names,
                weights=adapter_weights,
                normalized=normalize,
            )
        except Exception as e:
            logger.error("set_adapters_failed", error=str(e))
            raise InferenceError(f"Failed to set adapters: {e}", step="lora_composition")
    
    def load_adapter(self, adapter_path: Path, adapter_name: str) -> None:
        """
        Load a single LoRA adapter.
        
        Args:
            adapter_path: Path to the adapter file
            adapter_name: Name to register the adapter as
        """
        try:
            self._pipe.load_lora_weights(
                str(adapter_path.parent),
                weight_name=adapter_path.name,
                adapter_name=adapter_name,
            )
            logger.debug("adapter_loaded", name=adapter_name, path=str(adapter_path))
        except Exception as e:
            logger.error("adapter_load_failed", name=adapter_name, error=str(e))
            raise InferenceError(f"Failed to load adapter {adapter_name}: {e}", step="lora_loading")
    
    def generate(
        self,
        request: GenerationRequest,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> GenerationResult:
        """
        Generate an image from the request.
        
        Args:
            request: Generation request
            progress_callback: Optional callback(step, total_steps)
            
        Returns:
            Generation result with image and metadata
        """
        import time
        
        # Validate request
        request.validate()
        
        # Ensure pipeline is loaded
        if not self.is_loaded:
            if self.is_loading:
                raise InferenceError("Model is still loading... please wait.", step="model_loading")
            self.load()
        
        # Determine seed
        seed = request.seed
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
        
        # Create generator
        generator = torch.Generator(device=self._device_manager.device).manual_seed(seed)
        
        logger.info(
            "generation_started",
            prompt=request.prompt[:100],
            seed=seed,
            resolution=f"{request.width}x{request.height}",
            steps=request.steps,
        )
        
        start_time = time.time()
        
        try:
            # Build callback wrapper
            def step_callback(pipe: Any, step: int, timestep: Any, callback_kwargs: dict) -> dict:
                if progress_callback:
                    progress_callback(step, request.steps)
                return callback_kwargs
            
            # Generate
            if request.logo_image or request.background_image:
                image = self._generate_conditioned(request, generator, step_callback)
            else:
                result = self._pipe(
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    width=request.width,
                    height=request.height,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    generator=generator,
                    callback_on_step_end=step_callback if progress_callback else None,
                )
                image = result.images[0]
            generation_time = int((time.time() - start_time) * 1000)
            
            logger.info(
                "generation_completed",
                seed=seed,
                time_ms=generation_time,
            )
            
            return GenerationResult(
                image=image,
                seed=seed,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                width=request.width,
                height=request.height,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                adapters=request.adapters,
                adapter_weights=request.adapter_weights,
                generation_time_ms=generation_time,
            )
            
        except torch.cuda.OutOfMemoryError as e:
            self._device_manager.clear_cache()
            memory = self._device_manager.get_memory_info()
            raise OutOfMemoryError(
                available_gb=memory.get("free_gb"),
            )
        except Exception as e:
            logger.error("generation_failed", error=str(e))
            raise InferenceError(f"Generation failed: {e}", step="inference")
    
    def _generate_conditioned(
        self,
        request: GenerationRequest,
        generator: torch.Generator,
        callback: Optional[Callable] = None,
    ) -> Image.Image:
        """Generate image using logo and/or background conditioning."""
        from diffusers import StableDiffusionXLImg2ImgPipeline
        
        # 1. Prepare base image
        if request.background_image:
            base_image = request.background_image.resize((request.width, request.height), Image.LANCZOS)
        else:
            # Create a neutral gray/canvas background if none provided
            base_image = Image.new("RGB", (request.width, request.height), (200, 200, 200))

        # 2. Composite logo if provided
        if request.logo_image:
            logo = request.logo_image.convert("RGBA")
            # Scale logo to reasonable size (max 40% of canvas)
            max_logo_dim = int(min(request.width, request.height) * 0.4)
            logo.thumbnail((max_logo_dim, max_logo_dim), Image.LANCZOS)
            
            # Place in center (standard signboard location)
            offset = (
                (request.width - logo.width) // 2,
                (request.height - logo.height) // 2
            )
            
            # Create a simple white rectangular "backing" for the sign if we have a background
            if request.background_image:
                draw_img = Image.new("RGBA", (request.width, request.height), (0, 0, 0, 0))
                from PIL import ImageDraw
                padding = 20
                draw = ImageDraw.Draw(draw_img)
                draw.rectangle(
                    [
                        offset[0] - padding, offset[1] - padding,
                        offset[0] + logo.width + padding, offset[1] + logo.height + padding
                    ],
                    fill=(255, 255, 255, 255)
                )
                base_image.paste(draw_img, (0, 0), draw_img)

            base_image.paste(logo, offset, logo)

        # 3. Create Img2Img pipeline from current text2img pipeline
        # This sharing of components is efficient and recommended by diffusers
        img2img_pipe = StableDiffusionXLImg2ImgPipeline(**self._pipe.components)
        
        # 4. Run through model to homogenize lighting and texture
        # Lower strength (0.3-0.5) preserves the structure but blends the elements
        strength = request.logo_strength if request.logo_image else 0.4
        
        result = img2img_pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            image=base_image,
            strength=strength,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            generator=generator,
            callback_on_step_end=callback,
        )
        
        return result.images[0]

    def get_status(self) -> dict[str, Any]:
        """Get pipeline status."""
        memory = self._device_manager.get_memory_info()
        
        return {
            "loaded": self.is_loaded,
            "is_mock": False,
            "device": str(self._device_manager.device),
            "dtype": str(self._device_manager.dtype),
            "loaded_adapters": self._loaded_adapters,
            "gpu_memory": memory,
        }


_pipeline: Optional[SignForgePipeline] = None


def get_pipeline() -> SignForgePipeline:
    """Get the singleton pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = SignForgePipeline()
    return _pipeline
