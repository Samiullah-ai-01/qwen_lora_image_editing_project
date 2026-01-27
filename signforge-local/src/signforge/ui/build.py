"""
Build script for the React frontend.
"""

import os
import subprocess
from pathlib import Path
from signforge.core.config import get_config, get_project_root
from signforge.core.logging import get_logger

logger = get_logger(__name__)


def build_frontend() -> None:
    """Build the frontend application."""
    config = get_config()
    root = get_project_root()
    frontend_dir = root / "src" / "signforge" / "ui" / "frontend"
    static_dir = config.get_absolute_path(config.server.static_dir)
    
    logger.info("building_frontend", source=str(frontend_dir), dest=str(static_dir))
    
    # Check if node is available
    try:
        subprocess.run(["node", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("node_not_found", message="Node.js is required to build the frontend")
        raise RuntimeError("Node.js not found")
    
    # Install dependencies
    logger.info("installing_dependencies")
    subprocess.run(
        ["npm", "install"],
        cwd=frontend_dir,
        check=True,
        shell=True if os.name == "nt" else False,
    )
    
    # Build
    logger.info("running_build")
    subprocess.run(
        ["npm", "run", "build"],
        cwd=frontend_dir,
        check=True,
        shell=True if os.name == "nt" else False,
    )
    
    # Verify build output
    if not static_dir.exists() or not (static_dir / "index.html").exists():
        raise RuntimeError("Build failed: static directory or index.html missing")
    
    logger.info("frontend_build_complete")


if __name__ == "__main__":
    build_frontend()
