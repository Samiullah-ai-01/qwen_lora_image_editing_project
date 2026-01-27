"""
Prometheus metrics endpoint.
"""

from flask import Blueprint, Response
from signforge.monitoring.prometheus import get_metrics_output
from signforge.core.logging import get_logger

logger = get_logger(__name__)
bp = Blueprint("metrics", __name__)


@bp.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint."""
    output = get_metrics_output()
    return Response(output, mimetype="text/plain")
