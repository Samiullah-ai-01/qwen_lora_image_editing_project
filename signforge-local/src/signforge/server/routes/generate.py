"""
Generation endpoints.
"""

from flask import Blueprint, request, jsonify, send_file
from signforge.inference.service import get_service
from signforge.inference.api_schema import GenerateRequest
from signforge.core.errors import ValidationError, QueueFullError
from signforge.core.logging import get_logger
from pydantic import ValidationError as PydanticError

logger = get_logger(__name__)
bp = Blueprint("generate", __name__)


@bp.route("/generate", methods=["POST"])
def generate():
    """Submit a generation request."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Validate with Pydantic
        req = GenerateRequest(**data)
        
        # Submit to service
        service = get_service()
        result = service.submit(req.model_dump())
        
        return jsonify(result), 202
        
    except PydanticError as e:
        return jsonify({"error": "Validation error", "details": e.errors()}), 400
    except QueueFullError as e:
        return jsonify(e.to_dict()), 429
    except Exception as e:
        logger.error("generate_error", error=str(e))
        return jsonify({"error": str(e)}), 500


@bp.route("/generate/<item_id>", methods=["GET"])
def get_status(item_id: str):
    """Get status of a generation request."""
    service = get_service()
    status = service.get_status(item_id)
    
    if not status:
        return jsonify({"error": "Item not found"}), 404
    
    return jsonify(status)


@bp.route("/generate/<item_id>/result", methods=["GET"])
def get_result(item_id: str):
    """Get result of a completed generation."""
    service = get_service()
    result = service.get_result(item_id)
    
    if not result:
        status = service.get_status(item_id)
        if not status:
            return jsonify({"error": "Item not found"}), 404
        return jsonify({"error": "Result not ready", "status": status}), 202
    
    return jsonify(result)


@bp.route("/generate/<item_id>/image", methods=["GET"])
def get_image(item_id: str):
    """Get generated image directly."""
    service = get_service()
    result = service.get_result(item_id)
    
    if not result or "image_path" not in result:
        return jsonify({"error": "Image not available"}), 404
    
    return send_file(result["image_path"], mimetype="image/png")


@bp.route("/runs/<session_id>/images/<filename>")
def serve_image(session_id: str, filename: str):
    """Serve images from output directory."""
    from signforge.core.config import get_config
    from pathlib import Path
    
    # Validate filename to prevent traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return jsonify({"error": "Invalid filename"}), 400
    
    config = get_config()
    image_path = config.get_absolute_path(
        config.outputs.inference_runs_dir
    ) / session_id / "images" / filename
    
    if not image_path.exists():
        return jsonify({"error": "Image not found"}), 404
    
    return send_file(image_path, mimetype="image/png")


@bp.route("/queue", methods=["GET"])
def queue_status():
    """Get queue status."""
    service = get_service()
    return jsonify(service.get_queue_status())
