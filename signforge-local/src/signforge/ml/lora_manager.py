"""
LoRA Adapter Manager for SignForge.

Handles adapter discovery, loading, caching, and composition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from signforge.core.config import get_config
from signforge.core.logging import get_logger
from signforge.core.errors import LoRAError

logger = get_logger(__name__)


@dataclass
class AdapterInfo:
    """Information about a LoRA adapter."""
    
    name: str
    domain: str
    path: Path
    file_size: int
    
    # Metadata from training
    recommended_weight: float = 1.0
    training_run_id: Optional[str] = None
    training_dataset_hash: Optional[str] = None
    training_steps: Optional[int] = None
    training_loss: Optional[float] = None
    
    # Usage tracking
    load_count: int = 0
    last_used: Optional[datetime] = None
    
    # Compatibility
    compatible_with: list[str] = field(default_factory=list)
    conflicts_with: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "domain": self.domain,
            "path": str(self.path),
            "file_size": self.file_size,
            "file_size_mb": round(self.file_size / (1024 * 1024), 2),
            "recommended_weight": self.recommended_weight,
            "training_run_id": self.training_run_id,
            "training_dataset_hash": self.training_dataset_hash,
            "training_steps": self.training_steps,
            "training_loss": self.training_loss,
            "load_count": self.load_count,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "compatible_with": self.compatible_with,
            "conflicts_with": self.conflicts_with,
        }
    
    @classmethod
    def from_file(cls, path: Path, domain: str) -> "AdapterInfo":
        """Create adapter info from file path."""
        name = path.stem
        file_size = path.stat().st_size
        
        # Try to load metadata if exists
        metadata_path = path.with_suffix(".json")
        metadata = {}
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                pass
        
        return cls(
            name=name,
            domain=domain,
            path=path,
            file_size=file_size,
            recommended_weight=metadata.get("recommended_weight", 1.0),
            training_run_id=metadata.get("training_run_id"),
            training_dataset_hash=metadata.get("training_dataset_hash"),
            training_steps=metadata.get("training_steps"),
            training_loss=metadata.get("training_loss"),
            compatible_with=metadata.get("compatible_with", []),
            conflicts_with=metadata.get("conflicts_with", []),
        )


@dataclass
class CompositionProfile:
    """Profile for adapter composition weights."""
    
    name: str
    description: str
    weights: dict[str, float]
    normalize: bool = True
    
    @classmethod
    def default(cls) -> "CompositionProfile":
        """Get default composition profile."""
        return cls(
            name="default",
            description="Default adapter weights balanced for general use",
            weights={
                "sign_type": 1.0,
                "mounting": 0.9,
                "perspective": 0.7,
                "environment": 0.9,
                "lighting": 0.8,
                "material": 0.8,
            },
            normalize=True,
        )


class LoRAManager:
    """
    Manages LoRA adapter discovery, loading, and composition.
    
    Features:
    - Automatic adapter discovery
    - Registry with metadata
    - LRU caching for loaded adapters
    - Weight normalization
    - Conflict detection
    """
    
    _instance: Optional["LoRAManager"] = None
    
    def __new__(cls) -> "LoRAManager":
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the manager."""
        if self._initialized:
            return
        
        self._initialized = True
        self._config = get_config()
        
        # Registry of all known adapters
        self._registry: dict[str, AdapterInfo] = {}
        
        # Currently loaded adapter names
        self._loaded: set[str] = set()
        
        # Cache settings
        self._max_cached = self._config.lora.max_cached
        
        # Default composition profile
        self._default_weights = self._config.lora.default_weights
        
        # Scan for adapters
        self.scan_adapters()
    
    @property
    def loras_dir(self) -> Path:
        """Get the LoRAs directory."""
        return self._config.get_absolute_path(self._config.lora.base_dir)
    
    def scan_adapters(self) -> int:
        """
        Scan for available adapters.
        
        Returns:
            Number of adapters found
        """
        self._registry.clear()
        
        loras_dir = self.loras_dir
        if not loras_dir.exists():
            logger.warning("loras_dir_not_found", path=str(loras_dir))
            return 0
        
        count = 0
        
        # Scan each domain directory
        for domain_dir in loras_dir.iterdir():
            if not domain_dir.is_dir():
                continue
            
            domain = domain_dir.name
            
            # Scan for adapter files
            for adapter_file in domain_dir.glob("*.safetensors"):
                try:
                    info = AdapterInfo.from_file(adapter_file, domain)
                    full_name = f"{domain}/{info.name}"
                    self._registry[full_name] = info
                    count += 1
                except Exception as e:
                    logger.warning(
                        "adapter_scan_failed",
                        path=str(adapter_file),
                        error=str(e),
                    )
        
        # Save registry index
        self._save_registry_index()
        
        logger.info("adapters_scanned", count=count)
        return count
    
    def _save_registry_index(self) -> None:
        """Save the registry index to disk."""
        cache_dir = self._config.get_absolute_path(self._config.lora.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = cache_dir / "loras.json"
        
        index = {
            name: info.to_dict()
            for name, info in self._registry.items()
        }
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, default=str)
    
    def get_adapter(self, name: str) -> Optional[AdapterInfo]:
        """Get adapter info by name."""
        return self._registry.get(name)
    
    def get_adapters_by_domain(self, domain: str) -> list[AdapterInfo]:
        """Get all adapters for a domain."""
        return [
            info for info in self._registry.values()
            if info.domain == domain
        ]
    
    def list_domains(self) -> list[str]:
        """List all available domains."""
        return sorted(set(info.domain for info in self._registry.values()))
    
    def list_adapters(self) -> list[AdapterInfo]:
        """List all available adapters."""
        return list(self._registry.values())
    
    def get_registry_dict(self) -> dict[str, Any]:
        """Get the full registry as a dictionary."""
        domains = {}
        
        for name, info in self._registry.items():
            domain = info.domain
            if domain not in domains:
                domains[domain] = []
            domains[domain].append(info.to_dict())
        
        return {
            "domains": self.list_domains(),
            "adapters": domains,
            "total_count": len(self._registry),
        }
    
    def get_recommended_weights(
        self,
        adapter_names: list[str],
    ) -> list[float]:
        """
        Get recommended weights for adapters.
        
        Args:
            adapter_names: List of adapter names
            
        Returns:
            List of recommended weights
        """
        weights = []
        
        for name in adapter_names:
            info = self._registry.get(name)
            if info:
                # Use adapter-specific weight if available
                weight = info.recommended_weight
                # Fall back to domain default
                if weight == 1.0:
                    weight = self._default_weights.get(info.domain, 1.0)
            else:
                weight = 1.0
            weights.append(weight)
        
        return weights
    
    def normalize_weights(
        self,
        weights: list[float],
        method: str = "sum",
    ) -> list[float]:
        """
        Normalize adapter weights.
        
        Args:
            weights: List of weights
            method: Normalization method ('sum', 'max', 'none')
            
        Returns:
            Normalized weights
        """
        if not weights or method == "none":
            return weights
        
        if method == "sum":
            total = sum(abs(w) for w in weights)
            if total > 0:
                return [w / total for w in weights]
        elif method == "max":
            max_w = max(abs(w) for w in weights)
            if max_w > 0:
                return [w / max_w for w in weights]
        
        return weights
    
    def check_conflicts(
        self,
        adapter_names: list[str],
    ) -> list[tuple[str, str, str]]:
        """
        Check for conflicts between adapters.
        
        Args:
            adapter_names: List of adapter names to check
            
        Returns:
            List of (adapter1, adapter2, reason) tuples
        """
        conflicts = []
        
        for i, name1 in enumerate(adapter_names):
            info1 = self._registry.get(name1)
            if not info1:
                continue
                
            for name2 in adapter_names[i + 1:]:
                info2 = self._registry.get(name2)
                if not info2:
                    continue
                
                # Check explicit conflicts
                if name2 in info1.conflicts_with:
                    conflicts.append((name1, name2, "explicit_conflict"))
                elif name1 in info2.conflicts_with:
                    conflicts.append((name1, name2, "explicit_conflict"))
                
                # Check same domain (might cause interference)
                elif info1.domain == info2.domain:
                    conflicts.append((name1, name2, "same_domain"))
        
        return conflicts
    
    def prepare_adapters(
        self,
        adapter_names: list[str],
        weights: Optional[list[float]] = None,
        normalize: bool = True,
    ) -> tuple[list[str], list[float], list[Path]]:
        """
        Prepare adapters for loading.
        
        Args:
            adapter_names: Adapter names to prepare
            weights: Optional weights (uses recommended if not provided)
            normalize: Whether to normalize weights
            
        Returns:
            Tuple of (names, weights, paths)
        """
        if not adapter_names:
            return [], [], []
        
        # Validate all adapters exist
        paths = []
        valid_names = []
        
        for name in adapter_names:
            info = self._registry.get(name)
            if info is None:
                logger.warning("adapter_not_found", name=name)
                continue
            
            if not info.path.exists():
                logger.warning("adapter_file_missing", name=name, path=str(info.path))
                continue
            
            valid_names.append(name)
            paths.append(info.path)
        
        if not valid_names:
            raise LoRAError("No valid adapters found", details={"requested": adapter_names})
        
        # Get weights
        if weights is None:
            weights = self.get_recommended_weights(valid_names)
        else:
            # Ensure weights match valid adapters
            weights = weights[:len(valid_names)]
            # Pad if needed
            while len(weights) < len(valid_names):
                weights.append(1.0)
        
        # Normalize if requested
        if normalize and self._config.lora.normalize_weights:
            weights = self.normalize_weights(weights, method="sum")
        
        # Check for conflicts
        conflicts = self.check_conflicts(valid_names)
        for a1, a2, reason in conflicts:
            logger.warning(
                "adapter_conflict",
                adapter1=a1,
                adapter2=a2,
                reason=reason,
            )
        
        # Update usage stats
        for name in valid_names:
            info = self._registry[name]
            info.load_count += 1
            info.last_used = datetime.now()
        
        return valid_names, weights, paths
    
    def get_composition_suggestion(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        """
        Suggest adapter composition based on prompt analysis.
        
        Args:
            prompt: The generation prompt
            
        Returns:
            Suggested adapters with weights
        """
        suggestions = {
            "sign_type": None,
            "mounting": None,
            "perspective": None,
            "environment": None,
            "lighting": None,
        }
        
        prompt_lower = prompt.lower()
        
        # Sign type detection
        sign_keywords = {
            "channel": "channel_letters",
            "letter": "channel_letters",
            "box": "box_sign",
            "cabinet": "box_sign",
            "halo": "halo_lit",
            "backlit": "halo_lit",
            "blade": "blade",
            "flag": "blade",
            "monument": "monument",
            "ground": "monument",
            "pylon": "pylon",
            "pole": "pylon",
            "neon": "neon",
        }
        
        for keyword, concept in sign_keywords.items():
            if keyword in prompt_lower:
                full_name = f"sign_type/{concept}"
                if full_name in self._registry:
                    suggestions["sign_type"] = full_name
                    break
        
        # Environment detection
        env_keywords = {
            "urban": "urban_storefront",
            "city": "urban_storefront",
            "storefront": "urban_storefront",
            "mall": "mall_interior",
            "interior": "mall_interior",
            "indoor": "mall_interior",
            "night": "night",
            "evening": "night",
            "dark": "night",
            "day": "daytime",
            "daylight": "daytime",
            "sunny": "daytime",
        }
        
        for keyword, concept in env_keywords.items():
            if keyword in prompt_lower:
                full_name = f"environment/{concept}"
                if full_name in self._registry:
                    suggestions["environment"] = full_name
                    break
        
        # Build result
        adapters = []
        weights = []
        
        for domain, adapter_name in suggestions.items():
            if adapter_name:
                adapters.append(adapter_name)
                weights.append(self._default_weights.get(domain, 1.0))
        
        return {
            "adapters": adapters,
            "weights": weights,
            "suggestions": suggestions,
            "prompt_analyzed": prompt[:100],
        }


_manager: Optional[LoRAManager] = None


def get_lora_manager() -> LoRAManager:
    """Get the singleton LoRA manager instance."""
    global _manager
    if _manager is None:
        _manager = LoRAManager()
    return _manager
