"""
API documentation endpoint.
"""

from flask import Blueprint, jsonify
from signforge.core.config import get_config

bp = Blueprint("docs", __name__)


@bp.route("/docs", methods=["GET"])
def docs():
    """API documentation."""
    return jsonify({
        "name": "SignForge Local API",
        "version": "0.1.0",
        "description": "Local-first signage mockup generator with multi-LoRA composition",
        "endpoints": {
            "Generation": {
                "POST /generate": "Submit a generation request",
                "GET /generate/<item_id>": "Get request status",
                "GET /generate/<item_id>/result": "Get generation result",
                "GET /generate/<item_id>/image": "Get generated image",
                "GET /queue": "Get queue status",
            },
            "Adapters": {
                "GET /adapters": "List all adapters",
                "GET /adapters/<domain>": "List adapters by domain",
                "GET /adapters/<domain>/<name>": "Get specific adapter",
                "POST /adapters/suggest": "Suggest adapters for prompt",
                "POST /adapters/rescan": "Rescan adapter directory",
            },
            "Health": {
                "GET /health": "Health check",
                "GET /health/gpu": "GPU health details",
                "GET /health/ready": "Readiness check",
                "GET /health/live": "Liveness check",
            },
            "Monitoring": {
                "GET /metrics": "Prometheus metrics",
            },
        },
        "request_schema": {
            "prompt": "string (required)",
            "negative_prompt": "string",
            "width": "int (256-2048, default 1024)",
            "height": "int (256-2048, default 768)",
            "steps": "int (1-100, default 30)",
            "guidance_scale": "float (1-20, default 7.5)",
            "seed": "int (-1 for random)",
            "adapters": "list of adapter names",
            "adapter_weights": "list of weights",
            "normalize_weights": "bool (default true)",
        },
        "adapter_domains": [
            "sign_type",
            "mounting",
            "perspective",
            "environment",
            "lighting",
            "material",
        ],
    })
