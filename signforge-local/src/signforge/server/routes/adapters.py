"""
Adapter management endpoints.
"""

from flask import Blueprint, jsonify
from signforge.ml.lora_manager import get_lora_manager
from signforge.core.logging import get_logger

logger = get_logger(__name__)
bp = Blueprint("adapters", __name__)


@bp.route("/adapters", methods=["GET"])
def list_adapters():
    """List all available adapters."""
    manager = get_lora_manager()
    return jsonify(manager.get_registry_dict())


@bp.route("/adapters/<domain>", methods=["GET"])
def list_domain_adapters(domain: str):
    """List adapters for a specific domain."""
    manager = get_lora_manager()
    adapters = manager.get_adapters_by_domain(domain)
    return jsonify({
        "domain": domain,
        "adapters": [a.to_dict() for a in adapters],
        "count": len(adapters),
    })


@bp.route("/adapters/<domain>/<name>", methods=["GET"])
def get_adapter(domain: str, name: str):
    """Get specific adapter info."""
    manager = get_lora_manager()
    full_name = f"{domain}/{name}"
    adapter = manager.get_adapter(full_name)
    
    if not adapter:
        return jsonify({"error": "Adapter not found"}), 404
    
    return jsonify(adapter.to_dict())


@bp.route("/adapters/suggest", methods=["POST"])
def suggest_adapters():
    """Suggest adapters based on prompt."""
    from flask import request
    data = request.get_json() or {}
    prompt = data.get("prompt", "")
    
    if not prompt:
        return jsonify({"error": "Prompt required"}), 400
    
    manager = get_lora_manager()
    suggestions = manager.get_composition_suggestion(prompt)
    return jsonify(suggestions)


@bp.route("/adapters/rescan", methods=["POST"])
def rescan_adapters():
    """Rescan adapter directory."""
    manager = get_lora_manager()
    count = manager.scan_adapters()
    return jsonify({"message": "Scan complete", "count": count})


@bp.route("/adapters/weights/default", methods=["GET"])
def get_default_weights():
    """Get default adapter weights by domain."""
    from signforge.core.config import get_config
    config = get_config()
    return jsonify(config.lora.default_weights)
