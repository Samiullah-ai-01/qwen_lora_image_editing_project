"""
Training callbacks for monitoring and automation.
"""

from __future__ import annotations
from typing import Any, Dict, Optional

from signforge.core.logging import get_logger

logger = get_logger(__name__)


class TrainingCallback:
    """Base class for training callbacks."""
    def on_step_end(self, step: int, loss: float, **kwargs): pass
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]): pass
    def on_train_end(self, final_results: Dict[str, Any]): pass


class LoggingCallback(TrainingCallback):
    """Logs training progress to the console and logger."""
    
    def __init__(self, log_every: int = 10) -> None:
        self.log_every = log_every

    def on_step_end(self, step: int, loss: float, **kwargs):
        if step % self.log_every == 0:
            logger.info("train_step", step=step, loss=loss)


class CheckpointCallback(TrainingCallback):
    """Automatically saves checkpoints."""
    
    def __init__(self, manager: Any, save_every: int = 500) -> None:
        self.manager = manager
        self.save_every = save_every

    def on_step_end(self, step: int, loss: float, **kwargs):
        if step > 0 and step % self.save_every == 0:
            self.manager.save_checkpoint(
                kwargs.get("model"),
                kwargs.get("optimizer"),
                step,
                loss
            )


class CallbackManager:
    """Orchestrates multiple callbacks."""
    
    def __init__(self, callbacks: list[TrainingCallback]) -> None:
        self.callbacks = callbacks

    def on_step_end(self, step: int, loss: float, **kwargs):
        for cb in self.callbacks:
            cb.on_step_end(step, loss, **kwargs)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        for cb in self.callbacks:
            cb.on_epoch_end(epoch, metrics)

    def on_train_end(self, final_results: Dict[str, Any]):
        for cb in self.callbacks:
            cb.on_train_end(final_results)
