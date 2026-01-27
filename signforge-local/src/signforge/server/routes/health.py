"""
Health check endpoints.
"""

from flask import Blueprint, jsonify
from signforge.inference.service import get_service
from signforge.core.device import get_device_manager
from signforge.ml.pipeline import get_pipeline
from signforge.core.logging import get_logger

logger = get_logger(__name__)
bp = Blueprint("health", __name__)


@bp.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    device_manager = get_device_manager()
    pipeline = get_pipeline()
    
    memory = device_manager.get_memory_info()
    
    response = {
        "status": "healthy",
        "model_loaded": pipeline.is_loaded,
        "is_mock": pipeline.is_mock if pipeline.is_loaded else None,
        "device": str(device_manager.device),
        "dtype": str(device_manager.dtype),
        "gpu_available": device_manager.is_cuda_available,
    }
    
    if device_manager.is_cuda_available:
        response["gpu_memory_gb"] = memory.get("total_gb")
        response["gpu_used_gb"] = memory.get("allocated_gb")
        response["gpu_free_gb"] = memory.get("free_gb")
        response["gpu_utilization"] = memory.get("utilization_percent")
    
    try:
        service = get_service()
        queue_status = service.get_queue_status()
        response["queue_size"] = queue_status.get("queue_size", 0)
        response["queue_max"] = queue_status.get("max_size", 10)
        response["queue_running"] = queue_status.get("running", False)
    except:
        response["queue_size"] = 0
        response["queue_max"] = 10
        response["queue_running"] = False
    
    return jsonify(response)


@bp.route("/health/gpu", methods=["GET"])
def gpu_health():
    """Detailed GPU health."""
    device_manager = get_device_manager()
    
    if not device_manager.is_cuda_available:
        return jsonify({"error": "No GPU available"}), 503
    
    memory = device_manager.get_memory_info()
    gpu_info = device_manager.gpu_info
    
    return jsonify({
        "available": True,
        "name": gpu_info.name if gpu_info else "unknown",
        "compute_capability": gpu_info.compute_capability if gpu_info else None,
        "bf16_support": device_manager.is_bf16_supported,
        "memory": memory,
        "recommended_settings": device_manager.get_recommended_settings(),
    })


@bp.route("/health/ready", methods=["GET"])
def ready():
    """Readiness check for load balancers."""
    pipeline = get_pipeline()
    
    if not pipeline.is_loaded:
        return jsonify({"ready": False, "reason": "Model not loaded"}), 503
    
    return jsonify({"ready": True})


@bp.route("/health/live", methods=["GET"])
def live():
    """Liveness check."""
    return jsonify({"live": True})
