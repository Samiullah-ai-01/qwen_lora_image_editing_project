"""
Inference service orchestrating pipeline and queue.
"""

from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid
from PIL import Image
from signforge.core.config import get_config
from signforge.core.logging import get_logger, log_to_file
from signforge.ml.pipeline import SignForgePipeline, GenerationRequest, get_pipeline
from signforge.ml.lora_manager import get_lora_manager
from signforge.inference.queue import InferenceQueue, QueueItem

logger = get_logger(__name__)
_service: Optional[InferenceService] = None


def get_service() -> InferenceService:
    """Get the inference service singleton."""
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


class InferenceService:
    """High-level inference service."""

    def __init__(self) -> None:
        self._config = get_config()
        self._pipeline: Optional[SignForgePipeline] = None
        self._lora_manager = get_lora_manager()
        self._queue: Optional[InferenceQueue] = None
        self._session_id = str(uuid.uuid4())[:8]
        self._output_dir = self._config.get_absolute_path(
            self._config.outputs.inference_runs_dir
        ) / self._session_id
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def start(self, load_model: bool = True) -> None:
        """Start the inference service."""
        self._pipeline = get_pipeline()
        self._queue = InferenceQueue(worker_fn=self._process_item)
        self._queue.start()
        
        if load_model:
            import threading
            def load_task():
                try:
                    self._pipeline.load()
                    logger.info("service_fully_initialized")
                except Exception as e:
                    logger.error("delayed_load_failed", error=str(e))
            
            thread = threading.Thread(target=load_task, daemon=True)
            thread.start()
            logger.info("service_started_async_loading")
        else:
            logger.info("service_started_no_load")

    def stop(self) -> None:
        """Stop the service."""
        if self._queue:
            self._queue.stop()
        if self._pipeline:
            self._pipeline.unload()
        logger.info("service_stopped")

    def submit(self, request: dict) -> dict:
        """Submit a generation request."""
        if not self._queue:
            raise RuntimeError("Service not started")
        item = self._queue.submit(request)
        self._log_request(item.id, request)
        return {"item_id": item.id, "status": item.status.value}

    def get_status(self, item_id: str) -> Optional[dict]:
        """Get status of a request."""
        if not self._queue:
            return None
        item = self._queue.get_item(item_id)
        return item.to_dict() if item else None

    def get_result(self, item_id: str) -> Optional[dict]:
        """Get result of a completed request."""
        if not self._queue:
            return None
        item = self._queue.get_item(item_id)
        if not item or not item.result:
            return None
        return item.result

    def get_queue_status(self) -> dict:
        """Get queue status."""
        status = self._queue.get_status() if self._queue else {"running": False}
        status["session_id"] = self._session_id
        if self._pipeline:
            status["pipeline"] = self._pipeline.get_status()
        return status

    def _process_item(self, item: QueueItem) -> dict:
        """Process a queue item."""
        request = item.request
        
        def progress_callback(step: int, total: int) -> None:
            if self._queue:
                self._queue.set_progress(item.id, step, total)

        # Build generation request
        gen_request = GenerationRequest(
            prompt=request.get("prompt", ""),
            negative_prompt=request.get("negative_prompt", ""),
            width=request.get("width", 1024),
            height=request.get("height", 768),
            steps=request.get("steps", 30),
            guidance_scale=request.get("guidance_scale", 7.5),
            seed=request.get("seed", -1),
            adapters=request.get("adapters", []),
            adapter_weights=request.get("adapter_weights", []),
            normalize_weights=request.get("normalize_weights", True),
        )

        # Load adapters if needed
        if gen_request.adapters:
            names, weights, paths = self._lora_manager.prepare_adapters(
                gen_request.adapters,
                gen_request.adapter_weights,
                gen_request.normalize_weights,
            )
            for name, path in zip(names, paths):
                self._pipeline.load_adapter(path, name)
            self._pipeline.set_adapters(names, weights, normalize=False)

        # Generate
        result = self._pipeline.generate(gen_request, progress_callback)

        # Save image
        image_filename = f"{item.id}.png"
        image_path = self._output_dir / "images" / image_filename
        image_path.parent.mkdir(parents=True, exist_ok=True)
        result.image.save(image_path)

        # Log metadata
        metadata = result.to_dict()
        metadata["item_id"] = item.id
        metadata["image_path"] = str(image_path)
        self._log_metadata(metadata)

        return {
            "image_path": str(image_path),
            "image_url": f"/runs/{self._session_id}/images/{image_filename}",
            **metadata,
        }

    def _log_request(self, item_id: str, request: dict) -> None:
        """Log request to JSONL file."""
        log_to_file(
            {"item_id": item_id, "request": request},
            self._output_dir / "requests.jsonl",
        )

    def _log_metadata(self, metadata: dict) -> None:
        """Log generation metadata."""
        log_to_file(metadata, self._output_dir / "metadata.jsonl")

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def output_dir(self) -> Path:
        return self._output_dir
