"""
Diagnostic tools for assistant.
"""

from __future__ import annotations
import json
from signforge.core.device import get_device_manager
from signforge.ml.pipeline import get_pipeline
from signforge.ml.lora_manager import get_lora_manager
from signforge.core.config import get_config
from signforge.core.logging import get_logger

logger = get_logger(__name__)


class DiagnosticTools:
    """Tools for system diagnostics."""

    def diagnose(self) -> str:
        """Run full system diagnosis."""
        lines = ["🔍 **System Diagnosis**\n"]
        
        # Check GPU
        dm = get_device_manager()
        if dm.is_cuda_available:
            mem = dm.get_memory_info()
            lines.append(f"✓ GPU available: {dm.gpu_info.name if dm.gpu_info else 'Unknown'}")
            lines.append(f"  Memory: {mem.get('allocated_gb', 0):.1f}GB / {mem.get('total_gb', 0):.1f}GB used")
            if mem.get('utilization_percent', 0) > 90:
                lines.append("  ⚠️ High GPU memory usage - consider reducing resolution")
        else:
            lines.append("✗ No GPU detected - using CPU (slow)")
            lines.append("  Suggestion: Check CUDA installation")
        
        # Check model
        pipeline = get_pipeline()
        if pipeline.is_loaded:
            if pipeline.is_mock:
                lines.append("⚠ Model: Mock pipeline (no real generation)")
            else:
                lines.append("✓ Model loaded and ready")
        else:
            lines.append("✗ Model not loaded")
            lines.append("  Suggestion: Start the server or call load()")
        
        # Check adapters
        lora_mgr = get_lora_manager()
        adapters = lora_mgr.list_adapters()
        if adapters:
            lines.append(f"✓ Adapters found: {len(adapters)}")
        else:
            lines.append("⚠ No adapters found")
            lines.append("  Suggestion: Train adapters or check models/loras/ directory")
        
        # Check config
        try:
            config = get_config()
            lines.append("✓ Configuration loaded")
        except Exception as e:
            lines.append(f"✗ Configuration error: {e}")
        
        return "\n".join(lines)

    def explain_weights(self) -> str:
        """Explain current adapter weights."""
        config = get_config()
        weights = config.lora.default_weights
        
        lines = ["📊 **Adapter Weight Explanation**\n"]
        lines.append("Default weights by domain:")
        
        for domain, weight in weights.items():
            bar = "█" * int(weight * 10) + "░" * (10 - int(weight * 10))
            lines.append(f"  {domain}: {weight:.1f} [{bar}]")
        
        lines.append("\n**How weights work:**")
        lines.append("- Higher weight = stronger influence on output")
        lines.append("- Weights are normalized by default (sum to 1)")
        lines.append("- sign_type typically needs highest weight")
        lines.append("- perspective/lighting can use lower weights")
        
        lines.append("\n**Tips:**")
        lines.append("- If one adapter dominates, lower its weight")
        lines.append("- If effect is too subtle, increase weight")
        lines.append("- Disable normalization for manual control")
        
        return "\n".join(lines)

    def get_status(self) -> str:
        """Get current system status."""
        lines = ["📈 **Current Status**\n"]
        
        dm = get_device_manager()
        lines.append(f"Device: {dm.device}")
        lines.append(f"Dtype: {dm.dtype}")
        
        if dm.is_cuda_available:
            mem = dm.get_memory_info()
            lines.append(f"GPU Memory: {mem.get('free_gb', 0):.1f}GB free")
        
        pipeline = get_pipeline()
        lines.append(f"Pipeline loaded: {pipeline.is_loaded}")
        
        lora_mgr = get_lora_manager()
        lines.append(f"Available adapters: {len(lora_mgr.list_adapters())}")
        
        return "\n".join(lines)

    def get_last_error(self) -> str:
        """Get last error from logs."""
        config = get_config()
        log_dir = config.get_absolute_path(config.outputs.inference_runs_dir)
        
        # Find most recent session
        sessions = sorted(log_dir.iterdir(), reverse=True) if log_dir.exists() else []
        if not sessions:
            return "No recent sessions found"
        
        metadata_file = sessions[0] / "metadata.jsonl"
        if not metadata_file.exists():
            return "No metadata found in recent session"
        
        # Read last few entries
        lines = metadata_file.read_text().strip().split("\n")[-5:]
        errors = []
        for line in lines:
            try:
                data = json.loads(line)
                if "error" in data:
                    errors.append(data["error"])
            except:
                pass
        
        if errors:
            return f"Recent errors:\n" + "\n".join(f"- {e}" for e in errors)
        return "No recent errors found"
