"""
Inference queue for SignForge.
Bounded queue with worker thread for sequential processing.
"""

from __future__ import annotations
import queue
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
from signforge.core.config import get_config
from signforge.core.logging import get_logger
from signforge.core.errors import QueueFullError, TimeoutError

logger = get_logger(__name__)


class ItemStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QueueItem:
    """Item in the inference queue."""
    id: str
    request: dict
    status: ItemStatus = ItemStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    progress: int = 0
    total_steps: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "progress": self.progress,
            "total_steps": self.total_steps,
            "error": self.error,
        }


class InferenceQueue:
    """Bounded queue for inference requests."""

    def __init__(
        self,
        max_size: Optional[int] = None,
        timeout: Optional[int] = None,
        worker_fn: Optional[Callable[[QueueItem], Any]] = None,
    ) -> None:
        config = get_config()
        self.max_size = max_size or config.inference.max_queue_size
        self.timeout = timeout or config.inference.timeout_seconds
        self._queue: queue.Queue[QueueItem] = queue.Queue(maxsize=self.max_size)
        self._items: dict[str, QueueItem] = {}
        self._lock = threading.Lock()
        self._worker_fn = worker_fn
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._current_item: Optional[QueueItem] = None

    def start(self) -> None:
        """Start the worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        logger.info("queue_started", max_size=self.max_size)

    def stop(self) -> None:
        """Stop the worker thread."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        logger.info("queue_stopped")

    def submit(self, request: dict) -> QueueItem:
        """Submit a request to the queue."""
        if self._queue.full():
            raise QueueFullError(self._queue.qsize(), self.max_size)
        
        item = QueueItem(id=str(uuid.uuid4()), request=request, total_steps=request.get("steps", 30))
        with self._lock:
            self._items[item.id] = item
        self._queue.put(item)
        logger.debug("item_submitted", item_id=item.id)
        return item

    def get_item(self, item_id: str) -> Optional[QueueItem]:
        """Get item by ID."""
        with self._lock:
            return self._items.get(item_id)

    def get_status(self) -> dict:
        """Get queue status."""
        return {
            "running": self._running,
            "queue_size": self._queue.qsize(),
            "max_size": self.max_size,
            "current_item": self._current_item.id if self._current_item else None,
            "total_processed": sum(1 for i in self._items.values() 
                                   if i.status in (ItemStatus.COMPLETED, ItemStatus.FAILED)),
        }

    def _worker_loop(self) -> None:
        """Main worker loop."""
        while self._running:
            try:
                item = self._queue.get(timeout=1)
            except queue.Empty:
                continue

            self._current_item = item
            item.status = ItemStatus.PROCESSING
            item.started_at = datetime.now()
            logger.info("processing_item", item_id=item.id)

            try:
                if self._worker_fn:
                    result = self._worker_fn(item)
                    item.result = result
                    item.status = ItemStatus.COMPLETED
                else:
                    item.status = ItemStatus.FAILED
                    item.error = "No worker function configured"
            except Exception as e:
                logger.error("item_failed", item_id=item.id, error=str(e))
                item.status = ItemStatus.FAILED
                item.error = str(e)
            finally:
                item.completed_at = datetime.now()
                self._current_item = None
                self._queue.task_done()

    def set_progress(self, item_id: str, progress: int, total: int) -> None:
        """Update item progress."""
        with self._lock:
            item = self._items.get(item_id)
            if item:
                item.progress = progress
                item.total_steps = total

    def cleanup_old(self, max_age_seconds: int = 3600) -> int:
        """Remove old completed items."""
        now = datetime.now()
        removed = 0
        with self._lock:
            to_remove = [
                item_id for item_id, item in self._items.items()
                if item.status in (ItemStatus.COMPLETED, ItemStatus.FAILED)
                and item.completed_at
                and (now - item.completed_at).total_seconds() > max_age_seconds
            ]
            for item_id in to_remove:
                del self._items[item_id]
                removed += 1
        return removed
