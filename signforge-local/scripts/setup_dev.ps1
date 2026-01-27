$ErrorActionPreference = "Stop"

Write-Host "Setting up SignForge development environment..." -ForegroundColor Green

# Check for python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Python is required but not found."
    exit 1
}

# Check for node
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Warning "Node.js is required for the frontend but not found. Skipping frontend setup."
    $SkipFrontend = $true
}
else {
    $SkipFrontend = $false
}

# Create virtual environment
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Activate venv
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

# Install dependencies
Write-Host "Installing Python dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install frontend dependencies
if (-not $SkipFrontend) {
    Write-Host "Installing frontend dependencies..."
    Push-Location src\signforge\ui\frontend
    npm install
    Pop-Location
}
else {
    Write-Warning "Skipping frontend dependencies (Node.js missing)"
}

# Setup pre-commit
Write-Host "Setting up pre-commit hooks..."
pre-commit install

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "Run '.venv\Scripts\Activate.ps1' to start."
