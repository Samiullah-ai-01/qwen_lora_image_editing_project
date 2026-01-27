#!/bin/bash
# Run training for a concept

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <concept_name> [config_path]"
    exit 1
fi

CONCEPT=$1
CONFIG=${2:-"configs/training/base_lora.yaml"}

echo "Training LoRA for concept: $CONCEPT"
echo "Using config: $CONFIG"

# Activate venv if exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python -m signforge.training.train_lora \
    --config "$CONFIG" \
    --concept "$CONCEPT"
