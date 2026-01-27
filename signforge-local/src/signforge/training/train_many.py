"""
Batch training for multiple LoRA adapters.
"""

from __future__ import annotations
import argparse
import json
from datetime import datetime
from pathlib import Path
import yaml
from signforge.training.train_lora import train_lora
from signforge.core.config import get_config, get_project_root
from signforge.core.logging import get_logger

logger = get_logger(__name__)


def train_many(plan_path: Path, continue_on_error: bool = True) -> dict:
    """
    Train multiple LoRA adapters from a plan.
    
    Args:
        plan_path: Path to training plan YAML
        continue_on_error: Continue if one training fails
    
    Returns:
        Summary results
    """
    # Load plan
    with open(plan_path, "r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)
    
    domain = plan.get("domain", {}).get("name", "unknown")
    concepts = plan.get("concepts", [])
    base_config = plan.get("base_config", "base_lora.yaml")
    training_order = plan.get("training_order", [c["name"] for c in concepts])
    
    # Resolve base config path
    config_dir = plan_path.parent
    base_config_path = config_dir / base_config
    
    results = {
        "domain": domain,
        "plan_path": str(plan_path),
        "started_at": datetime.now().isoformat(),
        "concepts": {},
        "success_count": 0,
        "error_count": 0,
    }
    
    logger.info("batch_training_started", domain=domain, concepts=len(training_order))
    
    for concept_name in training_order:
        # Find concept config
        concept_config = next(
            (c for c in concepts if c["name"] == concept_name),
            {"name": concept_name}
        )
        
        full_concept = f"{domain}_{concept_name}"
        
        logger.info("training_concept", concept=full_concept)
        
        try:
            result = train_lora(
                config_path=base_config_path,
                concept=full_concept,
            )
            results["concepts"][concept_name] = {
                "status": "success",
                "result": result,
            }
            results["success_count"] += 1
            
        except Exception as e:
            logger.error("concept_training_failed", concept=full_concept, error=str(e))
            results["concepts"][concept_name] = {
                "status": "error",
                "error": str(e),
            }
            results["error_count"] += 1
            
            if not continue_on_error:
                break
    
    results["completed_at"] = datetime.now().isoformat()
    
    # Save summary
    app_config = get_config()
    summary_path = app_config.get_absolute_path(
        app_config.outputs.training_runs_dir
    ) / f"batch_{domain}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    
    logger.info(
        "batch_training_completed",
        success=results["success_count"],
        errors=results["error_count"],
    )
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train multiple LoRA adapters")
    parser.add_argument("--plan", type=Path, required=True, help="Training plan YAML")
    parser.add_argument("--no-continue", action="store_true", help="Stop on first error")
    args = parser.parse_args()
    
    train_many(args.plan, continue_on_error=not args.no_continue)


if __name__ == "__main__":
    main()
