param (
    [string]$Host = "0.0.0.0",
    [int]$Port = 8000
)

# Activate venv if exists
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

Write-Host "Starting SignForge server on ${Host}:${Port}..."
python -m signforge.server.app
