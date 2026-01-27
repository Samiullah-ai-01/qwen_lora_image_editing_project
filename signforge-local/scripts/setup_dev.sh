#!/bin/bash
# Setup development environment

set -e

# Colors
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${GREEN}Setting up SignForge development environment...${NC}"

# Check for python
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found."
    exit 1
fi

# Check for node
if ! command -v node &> /dev/null; then
    echo "Node.js is required for the frontend but not found. Skipping frontend setup."
    SKIP_FRONTEND=true
else
    SKIP_FRONTEND=false
fi

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install frontend dependencies
if [ "$SKIP_FRONTEND" = false ]; then
    echo "Installing frontend dependencies..."
    cd src/signforge/ui/frontend
    npm install
    cd ../../../..
else
    echo "Skipping frontend dependencies (Node.js missing)"
fi

# Setup pre-commit
echo "Setting up pre-commit hooks..."
pre-commit install

echo -e "${GREEN}Setup complete!${NC}"
echo "Run 'source .venv/bin/activate' to start."
