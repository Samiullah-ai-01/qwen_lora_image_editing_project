from flask import Blueprint, request, jsonify
from signforge.assistant.service import get_assistant
import time

bp = Blueprint("chat", __name__)

@bp.route("/chat", methods=["POST"])
def chat():
    """Chat with the SignForge Assistant."""
    try:
        data = request.get_json()
        if not data or "message" not in data:
            return jsonify({"error": "No message provided"}), 400
            
        message = data["message"]
        history = data.get("history", [])
        
        assistant = get_assistant()
        start_time = time.time()
        
        response = assistant.generate_response(message, history)
        
        return jsonify({
            "response": response,
            "latency_ms": int((time.time() - start_time) * 1000)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
