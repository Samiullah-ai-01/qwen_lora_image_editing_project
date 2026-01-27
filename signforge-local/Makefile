.PHONY: setup download-model preprocess train-one train-all serve test lint format clean help

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
VENV := venv
VENV_ACTIVATE := $(VENV)/Scripts/activate

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RESET := \033[0m

help: ## Show this help message
	@echo "SignForge Local - Make Targets"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'

setup: ## Set up the development environment
	@echo "$(GREEN)Setting up SignForge Local...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/pip install --upgrade pip
	$(VENV)/Scripts/pip install -r requirements.txt
	$(VENV)/Scripts/pip install -r requirements-dev.txt
	$(VENV)/Scripts/pip install -e .
	@echo "$(GREEN)Installing frontend dependencies...$(RESET)"
	cd src/signforge/ui/frontend && npm install
	@echo "$(GREEN)Setup complete!$(RESET)"
	@echo "Activate the virtual environment with: $(VENV_ACTIVATE)"

setup-linux: ## Set up on Linux/macOS
	@echo "$(GREEN)Setting up SignForge Local...$(RESET)"
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	$(VENV)/bin/pip install -r requirements-dev.txt
	$(VENV)/bin/pip install -e .
	@echo "$(GREEN)Installing frontend dependencies...$(RESET)"
	cd src/signforge/ui/frontend && npm install
	@echo "$(GREEN)Setup complete!$(RESET)"

download-model: ## Download the base Qwen-Image model
	@echo "$(GREEN)Downloading base model...$(RESET)"
	$(PYTHON) scripts/download_model.py

preprocess: ## Preprocess training data
	@echo "$(GREEN)Preprocessing data...$(RESET)"
	$(PYTHON) -m signforge.data.preprocess --input data/raw --output data/processed

train-one: ## Train a single LoRA adapter (use CONCEPT=name)
	@echo "$(GREEN)Training LoRA: $(CONCEPT)...$(RESET)"
	$(PYTHON) -m signforge.training.train_lora --config configs/training/base_lora.yaml --concept $(CONCEPT)

train-all: ## Train all LoRA adapters from plan
	@echo "$(GREEN)Training all LoRAs...$(RESET)"
	$(PYTHON) -m signforge.training.train_many --plan configs/training/sign_type.yaml

serve: ## Start the Flask server with built frontend
	@echo "$(GREEN)Building frontend...$(RESET)"
	cd src/signforge/ui/frontend && npm run build
	@echo "$(GREEN)Copying frontend build to static...$(RESET)"
	$(PYTHON) src/signforge/ui/build.py
	@echo "$(GREEN)Starting server...$(RESET)"
	$(PYTHON) -m signforge.server.app

serve-dev: ## Start server in development mode
	@echo "$(GREEN)Starting development server...$(RESET)"
	FLASK_DEBUG=1 $(PYTHON) -m signforge.server.app

ui-dev: ## Start frontend in development mode
	@echo "$(GREEN)Starting frontend dev server...$(RESET)"
	cd src/signforge/ui/frontend && npm run dev

test: ## Run all tests
	@echo "$(GREEN)Running tests...$(RESET)"
	pytest tests/ -v --tb=short

test-unit: ## Run unit tests only
	pytest tests/unit -v

test-integration: ## Run integration tests only
	pytest tests/integration -v

test-e2e: ## Run end-to-end tests
	pytest tests/e2e -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=src/signforge --cov-report=html --cov-report=term

lint: ## Run linting
	@echo "$(GREEN)Running linters...$(RESET)"
	ruff check src/ tests/
	mypy src/signforge

format: ## Format code
	@echo "$(GREEN)Formatting code...$(RESET)"
	black src/ tests/
	ruff check --fix src/ tests/

check: lint test ## Run linting and tests

logs: ## Tail recent logs
	$(PYTHON) scripts/tail_logs.py

clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-outputs: ## Clean generated outputs (careful!)
	@echo "$(YELLOW)Warning: This will delete all generated outputs!$(RESET)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf outputs/training_runs/*
	rm -rf outputs/inference_runs/*

docker-monitoring: ## Start monitoring stack (Prometheus + Grafana)
	@echo "$(GREEN)Starting monitoring stack...$(RESET)"
	cd src/signforge/monitoring/dashboard && docker-compose -f docker-compose.monitoring.yaml up -d

docker-monitoring-down: ## Stop monitoring stack
	cd src/signforge/monitoring/dashboard && docker-compose -f docker-compose.monitoring.yaml down

build-frontend: ## Build frontend for production
	cd src/signforge/ui/frontend && npm run build
	$(PYTHON) src/signforge/ui/build.py

install-playwright: ## Install Playwright browsers for E2E testing
	playwright install chromium

validate-config: ## Validate all configuration files
	$(PYTHON) -c "from signforge.core.config import validate_all_configs; validate_all_configs()"
