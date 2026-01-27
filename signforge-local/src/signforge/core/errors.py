"""
Custom exception classes for SignForge.

Provides structured error handling with context and recovery suggestions.
"""

from __future__ import annotations

from typing import Any, Optional


class SignForgeError(Exception):
    """Base exception for all SignForge errors."""
    
    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
        suggestion: Optional[str] = None,
    ) -> None:
        """
        Initialize the error.
        
        Args:
            message: Human-readable error message
            code: Machine-readable error code
            details: Additional context as a dict
            suggestion: Suggested fix or next step
        """
        super().__init__(message)
        self.message = message
        self.code = code or "SIGNFORGE_ERROR"
        self.details = details or {}
        self.suggestion = suggestion
    
    def to_dict(self) -> dict[str, Any]:
        """Convert error to a dictionary for JSON serialization."""
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
            "suggestion": self.suggestion,
        }
    
    def __str__(self) -> str:
        """String representation."""
        parts = [self.message]
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)


class ConfigError(SignForgeError):
    """Configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize config error."""
        details = kwargs.pop("details", {})
        if config_path:
            details["config_path"] = config_path
        if key:
            details["key"] = key
        
        super().__init__(
            message=message,
            code="CONFIG_ERROR",
            details=details,
            **kwargs,
        )


class ModelError(SignForgeError):
    """Model loading and execution errors."""
    
    def __init__(
        self,
        message: str,
        model_path: Optional[str] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize model error."""
        details = kwargs.pop("details", {})
        if model_path:
            details["model_path"] = model_path
        if model_id:
            details["model_id"] = model_id
        
        super().__init__(
            message=message,
            code="MODEL_ERROR",
            details=details,
            **kwargs,
        )


class LoRAError(SignForgeError):
    """LoRA adapter-related errors."""
    
    def __init__(
        self,
        message: str,
        adapter_name: Optional[str] = None,
        adapter_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize LoRA error."""
        details = kwargs.pop("details", {})
        if adapter_name:
            details["adapter_name"] = adapter_name
        if adapter_path:
            details["adapter_path"] = adapter_path
        
        super().__init__(
            message=message,
            code="LORA_ERROR",
            details=details,
            **kwargs,
        )


class InferenceError(SignForgeError):
    """Inference-related errors."""
    
    def __init__(
        self,
        message: str,
        request_id: Optional[str] = None,
        step: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize inference error."""
        details = kwargs.pop("details", {})
        if request_id:
            details["request_id"] = request_id
        if step:
            details["step"] = step
        
        super().__init__(
            message=message,
            code="INFERENCE_ERROR",
            details=details,
            **kwargs,
        )


class TrainingError(SignForgeError):
    """Training-related errors."""
    
    def __init__(
        self,
        message: str,
        run_id: Optional[str] = None,
        step: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize training error."""
        details = kwargs.pop("details", {})
        if run_id:
            details["run_id"] = run_id
        if step is not None:
            details["step"] = step
        
        super().__init__(
            message=message,
            code="TRAINING_ERROR",
            details=details,
            **kwargs,
        )


class DataError(SignForgeError):
    """Data loading and processing errors."""
    
    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize data error."""
        details = kwargs.pop("details", {})
        if file_path:
            details["file_path"] = file_path
        
        super().__init__(
            message=message,
            code="DATA_ERROR",
            details=details,
            **kwargs,
        )


class ValidationError(SignForgeError):
    """Input validation errors."""
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize validation error."""
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate long values
        
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            details=details,
            **kwargs,
        )


class QueueFullError(SignForgeError):
    """Queue is full, cannot accept more requests."""
    
    def __init__(
        self,
        queue_size: int,
        max_size: int,
        **kwargs: Any,
    ) -> None:
        """Initialize queue full error."""
        super().__init__(
            message=f"Request queue is full ({queue_size}/{max_size})",
            code="QUEUE_FULL",
            details={"queue_size": queue_size, "max_size": max_size},
            suggestion="Wait for current requests to complete or reduce request rate",
            **kwargs,
        )


class TimeoutError(SignForgeError):
    """Operation timed out."""
    
    def __init__(
        self,
        operation: str,
        timeout_seconds: float,
        **kwargs: Any,
    ) -> None:
        """Initialize timeout error."""
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout_seconds}s",
            code="TIMEOUT_ERROR",
            details={"operation": operation, "timeout_seconds": timeout_seconds},
            suggestion="Try reducing resolution or complexity, or increase timeout",
            **kwargs,
        )


class GPUError(SignForgeError):
    """GPU-related errors (OOM, not available, etc.)."""
    
    def __init__(
        self,
        message: str,
        gpu_index: Optional[int] = None,
        memory_info: Optional[dict] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize GPU error."""
        details = kwargs.pop("details", {})
        if gpu_index is not None:
            details["gpu_index"] = gpu_index
        if memory_info:
            details["memory_info"] = memory_info
        
        super().__init__(
            message=message,
            code="GPU_ERROR",
            details=details,
            **kwargs,
        )


class OutOfMemoryError(GPUError):
    """GPU out of memory."""
    
    def __init__(
        self,
        required_gb: Optional[float] = None,
        available_gb: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize OOM error."""
        message = "GPU out of memory"
        if required_gb and available_gb:
            message = f"GPU out of memory: need {required_gb:.1f}GB, have {available_gb:.1f}GB"
        
        details = {}
        if required_gb:
            details["required_gb"] = required_gb
        if available_gb:
            details["available_gb"] = available_gb
        
        super().__init__(
            message=message,
            memory_info=details,
            suggestion="Reduce resolution, enable VAE tiling, or use attention slicing",
            **kwargs,
        )


class SafetyError(SignForgeError):
    """Safety validation errors (blocked content, etc.)."""
    
    def __init__(
        self,
        message: str,
        reason: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize safety error."""
        details = kwargs.pop("details", {})
        if reason:
            details["reason"] = reason
        
        super().__init__(
            message=message,
            code="SAFETY_ERROR",
            details=details,
            **kwargs,
        )
