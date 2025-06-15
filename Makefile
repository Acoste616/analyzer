.PHONY: help install test run docker-up docker-down clean lint format check setup dev

# Default target
help:
	@echo "🚀 Jarvis EDU Extractor - Available Commands:"
	@echo ""
	@echo "📦 Installation & Setup:"
	@echo "  make install     - Install dependencies"
	@echo "  make setup       - Complete project setup"
	@echo "  make dev         - Install development dependencies"
	@echo ""
	@echo "🐳 Docker Commands:"
	@echo "  make docker-up   - Start all Docker services"
	@echo "  make docker-down - Stop all Docker services"
	@echo "  make docker-build - Build Docker images"
	@echo "  make docker-logs - Show Docker logs"
	@echo ""
	@echo "🧪 Testing & Quality:"
	@echo "  make test        - Run tests"
	@echo "  make test-cov    - Run tests with coverage report"
	@echo "  make lint        - Run linting checks"
	@echo "  make format      - Format code"
	@echo "  make check       - Run all quality checks"
	@echo ""
	@echo "🏃 Running:"
	@echo "  make run         - Run CLI help"
	@echo "  make serve       - Start web server"
	@echo "  make status      - Check system status"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  make clean       - Clean temporary files"
	@echo "  make clean-all   - Clean everything including Docker"
	@echo "  make update      - Update dependencies"

# Installation
install:
	@echo "📦 Installing dependencies..."
	pip install -e .
	@echo "✅ Installation complete!"

dev:
	@echo "📦 Installing development dependencies..."
	pip install -e .[dev]
	@echo "✅ Development setup complete!"

setup: install
	@echo "🔧 Setting up project..."
	# Create directories
	mkdir -p input output models cache logs temp data
	# Copy example config
	cp env.example .env 2>/dev/null || true
	# Download spaCy models
	python -c "import spacy; spacy.cli.download('en_core_web_sm')" || true
	python -c "import spacy; spacy.cli.download('pl_core_news_sm')" || true
	# Install Playwright browsers
	playwright install chromium --with-deps || true
	@echo "✅ Project setup complete!"

# Docker commands
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose build

docker-up:
	@echo "🐳 Starting Docker services..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "🌐 Web UI: http://localhost:8000"
	@echo "📊 Flower (Celery): http://localhost:5555"
	@echo "🔍 Qdrant: http://localhost:6333"

docker-down:
	@echo "🐳 Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped!"

docker-logs:
	@echo "📋 Showing Docker logs..."
	docker-compose logs -f

docker-clean:
	@echo "🧹 Cleaning Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f

# Testing
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v

test-cov:
	@echo "🧪 Running tests with coverage..."
	pytest tests/ -v --cov=jarvis_edu --cov-report=html --cov-report=term
	@echo "📊 Coverage report: htmlcov/index.html"

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/ -v -m "unit"

test-integration:
	@echo "🧪 Running integration tests..."
	pytest tests/ -v -m "integration"

# Code quality
lint:
	@echo "🔍 Running linting checks..."
	flake8 jarvis_edu tests
	mypy jarvis_edu
	bandit -r jarvis_edu

format:
	@echo "🎨 Formatting code..."
	black jarvis_edu tests
	isort jarvis_edu tests

check: lint test
	@echo "✅ All quality checks passed!"

# Running
run:
	@echo "🏃 Running Jarvis EDU CLI..."
	python -m jarvis_edu.cli --help

serve:
	@echo "🌐 Starting web server..."
	python -m jarvis_edu.cli serve

status:
	@echo "📊 Checking system status..."
	python -m jarvis_edu.cli status

extract:
	@echo "📄 Example: Extract from video file"
	@echo "Usage: python -m jarvis_edu.cli extract path/to/video.mp4 -o output/"

# Maintenance
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov .pytest_cache .mypy_cache 2>/dev/null || true
	rm -rf build dist 2>/dev/null || true
	@echo "✅ Cleanup complete!"

clean-all: clean docker-clean
	@echo "🧹 Deep cleaning..."
	rm -rf data cache logs temp models 2>/dev/null || true
	@echo "✅ Deep cleanup complete!"

update:
	@echo "📦 Updating dependencies..."
	pip install --upgrade pip
	pip install -e .[dev] --upgrade
	@echo "✅ Dependencies updated!"

# Development shortcuts
dev-setup: dev setup
	@echo "🚀 Development environment ready!"

quick-test:
	@echo "⚡ Quick test run..."
	pytest tests/ -x --ff

watch-tests:
	@echo "👀 Watching tests..."
	pytest-watch tests/

# Production helpers
prod-check:
	@echo "🔒 Production readiness check..."
	@echo "⚠️  Remember to:"
	@echo "   1. Set SECRET_KEY in production"
	@echo "   2. Configure proper database URLs"
	@echo "   3. Set up monitoring (Sentry DSN)"
	@echo "   4. Review security settings"

# Ollama helpers
ollama-pull:
	@echo "🤖 Pulling Ollama models..."
	docker exec jarvis-ollama ollama pull llama3
	docker exec jarvis-ollama ollama pull codellama

ollama-list:
	@echo "📋 Available Ollama models:"
	docker exec jarvis-ollama ollama list

# Database helpers
db-init:
	@echo "🗄️ Initializing database..."
	python -c "from jarvis_edu.storage.database import init_db; init_db()"

db-reset:
	@echo "🗄️ Resetting database..."
	rm -f data/jarvis_edu.db
	$(MAKE) db-init 