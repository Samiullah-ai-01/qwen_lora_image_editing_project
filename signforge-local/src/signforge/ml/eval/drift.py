"""
Drift detection for SignForge model monitoring.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import json
from pathlib import Path
from signforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DriftSnapshot:
    """Snapshot of distribution at a point in time."""
    timestamp: datetime
    adapter_usage: dict[str, int] = field(default_factory=dict)
    prompt_length_mean: float = 0.0
    resolution_dist: dict[str, int] = field(default_factory=dict)
    failure_rate: float = 0.0
    sample_count: int = 0


class DriftDetector:
    """Detects distribution drift over time."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.current_window: list[dict] = []
        self.snapshots: list[DriftSnapshot] = []

    def record(self, request_data: dict) -> None:
        """Record a request for drift analysis."""
        self.current_window.append({
            "timestamp": datetime.now().isoformat(),
            "adapters": request_data.get("adapters", []),
            "prompt_length": len(request_data.get("prompt", "")),
            "resolution": f"{request_data.get('width', 0)}x{request_data.get('height', 0)}",
            "success": request_data.get("success", True),
        })
        if len(self.current_window) >= self.window_size:
            self._create_snapshot()

    def _create_snapshot(self) -> None:
        """Create snapshot from current window."""
        if not self.current_window:
            return
        adapter_usage: dict[str, int] = {}
        resolution_dist: dict[str, int] = {}
        prompt_lengths = []
        failures = 0

        for r in self.current_window:
            for a in r.get("adapters", []):
                adapter_usage[a] = adapter_usage.get(a, 0) + 1
            res = r.get("resolution", "unknown")
            resolution_dist[res] = resolution_dist.get(res, 0) + 1
            prompt_lengths.append(r.get("prompt_length", 0))
            if not r.get("success", True):
                failures += 1

        snapshot = DriftSnapshot(
            timestamp=datetime.now(),
            adapter_usage=adapter_usage,
            prompt_length_mean=sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
            resolution_dist=resolution_dist,
            failure_rate=failures / len(self.current_window),
            sample_count=len(self.current_window),
        )
        self.snapshots.append(snapshot)
        self.current_window.clear()
        logger.info("drift_snapshot_created", sample_count=snapshot.sample_count)

    def check_drift(self, threshold: float = 0.3) -> list[dict]:
        """Check for significant drift between recent snapshots."""
        alerts = []
        if len(self.snapshots) < 2:
            return alerts
        prev, curr = self.snapshots[-2], self.snapshots[-1]
        
        # Check failure rate drift
        if curr.failure_rate - prev.failure_rate > threshold:
            alerts.append({
                "type": "failure_rate_increase",
                "previous": prev.failure_rate,
                "current": curr.failure_rate,
            })
        
        # Check adapter usage shift
        all_adapters = set(prev.adapter_usage.keys()) | set(curr.adapter_usage.keys())
        for adapter in all_adapters:
            prev_pct = prev.adapter_usage.get(adapter, 0) / max(prev.sample_count, 1)
            curr_pct = curr.adapter_usage.get(adapter, 0) / max(curr.sample_count, 1)
            if abs(curr_pct - prev_pct) > threshold:
                alerts.append({
                    "type": "adapter_usage_shift",
                    "adapter": adapter,
                    "change": curr_pct - prev_pct,
                })
        return alerts

    def get_summary(self) -> dict:
        """Get drift detection summary."""
        return {
            "total_snapshots": len(self.snapshots),
            "current_window_size": len(self.current_window),
            "latest_alerts": self.check_drift() if len(self.snapshots) >= 2 else [],
        }

    def save(self, path: Path) -> None:
        """Save snapshots to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [{
            "timestamp": s.timestamp.isoformat(),
            "adapter_usage": s.adapter_usage,
            "prompt_length_mean": s.prompt_length_mean,
            "failure_rate": s.failure_rate,
            "sample_count": s.sample_count,
        } for s in self.snapshots]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
