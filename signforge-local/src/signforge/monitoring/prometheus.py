"""
Prometheus metrics for SignForge.
"""

from __future__ import annotations
from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
from signforge.core.logging import get_logger

logger = get_logger(__name__)

# Request metrics
REQUESTS_TOTAL = Counter(
    "signforge_requests_total",
    "Total generation requests",
    ["status"]
)

REQUEST_LATENCY = Histogram(
    "signforge_request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120, 300]
)

# Queue metrics
QUEUE_DEPTH = Gauge(
    "signforge_queue_depth",
    "Current queue depth"
)

QUEUE_MAX = Gauge(
    "signforge_queue_max",
    "Maximum queue size"
)

# GPU metrics
GPU_MEMORY_BYTES = Gauge(
    "signforge_gpu_memory_bytes",
    "GPU memory usage in bytes",
    ["type"]
)

# Adapter metrics
ADAPTER_USAGE = Counter(
    "signforge_adapter_usage_total",
    "Adapter usage count",
    ["adapter", "domain"]
)

# Generation metrics
GENERATION_STEPS = Histogram(
    "signforge_generation_steps",
    "Number of generation steps",
    buckets=[10, 15, 20, 30, 50, 75, 100]
)

GENERATION_RESOLUTION = Counter(
    "signforge_generation_resolution_total",
    "Generation resolution distribution",
    ["resolution"]
)


def track_request(status: str, latency_seconds: float) -> None:
    """Track a request."""
    REQUESTS_TOTAL.labels(status=status).inc()
    REQUEST_LATENCY.observe(latency_seconds)


def track_generation(
    steps: int,
    width: int,
    height: int,
    adapters: list[str],
) -> None:
    """Track generation metrics."""
    GENERATION_STEPS.observe(steps)
    GENERATION_RESOLUTION.labels(resolution=f"{width}x{height}").inc()
    
    for adapter in adapters:
        parts = adapter.split("/")
        domain = parts[0] if len(parts) > 1 else "unknown"
        ADAPTER_USAGE.labels(adapter=adapter, domain=domain).inc()


def update_queue_metrics(current: int, max_size: int) -> None:
    """Update queue metrics."""
    QUEUE_DEPTH.set(current)
    QUEUE_MAX.set(max_size)


def update_gpu_metrics(allocated: int, total: int) -> None:
    """Update GPU memory metrics."""
    GPU_MEMORY_BYTES.labels(type="allocated").set(allocated)
    GPU_MEMORY_BYTES.labels(type="total").set(total)
    GPU_MEMORY_BYTES.labels(type="free").set(total - allocated)


def get_metrics_output() -> str:
    """Get Prometheus metrics output."""
    return generate_latest(REGISTRY).decode("utf-8")
