param (
    [Parameter(Mandatory=$true)]
    [string]$Concept,
    
    [string]$Config = "configs/training/base_lora.yaml"
)

Write-Host "Training LoRA for concept: $Concept"
Write-Host "Using config: $Config"

# Activate venv if exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

python -m signforge.training.train_lora --config "$Config" --concept "$Concept"
