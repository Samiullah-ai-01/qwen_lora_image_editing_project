"""
Flask application factory for SignForge.
"""

from __future__ import annotations
import os
from pathlib import Path
from flask import Flask
from flask_cors import CORS
from signforge.core.config import get_config
from signforge.core.logging import configure_logging, get_logger
from signforge.inference.service import InferenceService, get_service

logger = get_logger(__name__)


def create_app(test_mode: bool = False) -> Flask:
    """Create and configure the Flask app."""
    config = get_config()
    
    # Configure logging
    if not test_mode:
        log_file = config.get_absolute_path(config.outputs.inference_runs_dir) / "logs" / "app.log"
        configure_logging(
            level=config.logging.level,
            format=config.logging.format,
            log_file=log_file,
        )
    
    # Create Flask app
    app = Flask(
        __name__,
        static_folder=str(config.get_absolute_path(config.server.static_dir)),
        static_url_path="",
    )
    
    # Configure app
    app.config["MAX_CONTENT_LENGTH"] = config.server.max_content_length
    app.config["JSON_SORT_KEYS"] = False
    
    # Enable CORS
    if config.server.cors_enabled:
        CORS(app, origins=config.server.cors_origins)
    
    # Register blueprints
    from signforge.server.routes.generate import bp as generate_bp
    from signforge.server.routes.adapters import bp as adapters_bp
    from signforge.server.routes.health import bp as health_bp
    from signforge.server.routes.metrics import bp as metrics_bp
    from signforge.server.routes.docs import bp as docs_bp
    
    app.register_blueprint(generate_bp)
    app.register_blueprint(adapters_bp)
    app.register_blueprint(health_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(docs_bp)
    
    # Index route - serve frontend
    @app.route("/")
    def index():
        return app.send_static_file("index.html")
    
    # Error handlers
    @app.errorhandler(404)
    def not_found(e):
        # Try to serve index.html for SPA routing
        try:
            return app.send_static_file("index.html")
        except:
            return {"error": "Not found"}, 404
    
    @app.errorhandler(500)
    def server_error(e):
        logger.error("server_error", error=str(e))
        return {"error": "Internal server error"}, 500
    
    @app.before_request
    def init_service():
        service = get_service()
        if not getattr(service, "_started", False):
            service.start()
            service._started = True
    
    logger.info("app_created", test_mode=test_mode)
    return app


def run_server(host: str = None, port: int = None, debug: bool = False) -> None:
    """Run the Flask server."""
    config = get_config()
    host = host or config.server.host
    port = port or config.server.port
    
    app = create_app()
    
    logger.info("starting_server", host=host, port=port)
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    run_server()
