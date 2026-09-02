# Makefile for AI Spectrum Management for NTN Project

.PHONY: help install quickstart pipeline-demo train-models benchmark compile-workflow launch-autoresearch clean docs test

# Default target
help:
	@echo "🚀 AI Spectrum Management for Non-terrestrial Networks (NTN)"
	@echo ""
	@echo "Available targets:"
	@echo ""
	@echo "  Installation & Setup:"
	@echo "    make install              - Install all dependencies via uv sync"
	@echo "    make setup                - Setup data directories and configs"
	@echo ""
	@echo "  Quick Start:"
	@echo "    make quickstart           - Run minimal end-to-end demo"
	@echo ""
	@echo "  Pipeline Execution:"
	@echo "    make pipeline-demo        - Run full 10-step pipeline"
	@echo "    make dataset              - Step 1: Generate raw dataset"
	@echo "    make process-data         - Step 2: Process data to learning states"
	@echo "    make ground-truth         - Step 3: Generate optimal labels"
	@echo "    make train-models         - Step 4: Train neural models"
	@echo "    make build-agent          - Step 5: Build AI agent"
	@echo "    make compile-workflow     - Step 7: Compile NemoIR workflow"
	@echo "    make benchmark            - Step 8: Run benchmarks"
	@echo "    make online-demo          - Step 9: Online execution demo"
	@echo "    make launch-autoresearch  - Step 10: Launch autonomous research"
	@echo ""
	@echo "  Development:"
	@echo "    make test                 - Run test suite"
	@echo "    make lint                 - Run linting (ruff, black, mypy)"
	@echo "    make format               - Format code (black, ruff)"
	@echo "    make docs                 - Build documentation"
	@echo ""
	@echo "  Cleanup:"
	@echo "    make clean                - Remove build artifacts, cache, logs"
	@echo "    make clean-data           - Remove generated datasets"
	@echo ""

# Installation
install:
	@echo "📦 Installing dependencies..."
	uv sync
	@echo "✅ Installation complete!"

setup:
	@echo "🔧 Setting up project structure..."
	mkdir -p data/raw data/processed data/ground_truth data/models data/indices
	mkdir -p experiments/baseline_models experiments/adversarial_water_filling experiments/diffract_optimization
	mkdir -p experiments/rl_agent experiments/llm_agent experiments/nemoir_compiled experiments/autoresearch_logs
	mkdir -p research/experiment_logs logs notebooks
	@echo "✅ Project setup complete!"

# Quick Start
quickstart: install setup
	@echo "⚡ Running quick-start demo (minimal dataset + single model training)..."
	uv run src/cli.py generate-dataset --num-scenarios 10 --output data/raw/
	uv run src/cli.py process-data --input data/raw/ --output data/processed/
	uv run src/cli.py generate-labels --method adversarial_wf --num-scenarios 10 --output data/ground_truth/
	uv run src/cli.py train-model --model supervised --epochs 10 --gpu 0
	@echo "✅ Quick-start complete!"

# Full Pipeline
pipeline-demo: install setup
	@echo "🚀 Running full 10-step pipeline demo..."
	uv run src/cli.py pipeline --stages 1,2,3,4,5,7,8
	@echo "✅ Pipeline demo complete!"

# Individual Steps
dataset:
	@echo "Step 1️⃣  Generating NTN dataset..."
	uv run src/cli.py generate-dataset --num-scenarios 100 --output data/raw/

process-data:
	@echo "Step 2️⃣  Processing data to learning states..."
	uv run src/cli.py process-data --input data/raw/ --output data/processed/

ground-truth:
	@echo "Step 3️⃣  Generating ground-truth optimal labels..."
	uv run src/cli.py generate-labels --method adversarial_wf --num-scenarios 100 --output data/ground_truth/

train-models: train-supervised train-rl train-predictors
	@echo "Step 4️⃣  All models trained!"

train-supervised:
	@echo "  → Training supervised (imitation) model..."
	uv run src/cli.py train-model --model supervised --epochs 100 --gpu 0

train-rl:
	@echo "  → Training RL agent (PPO)..."
	uv run src/cli.py train-model --model rl --epochs 1000 --gpu 0

train-predictors:
	@echo "  → Training channel/traffic/jammer predictors..."
	uv run src/cli.py train-model --model channel_predictor --epochs 50 --gpu 0
	uv run src/cli.py train-model --model jammer_detector --epochs 50 --gpu 0

build-agent:
	@echo "Step 5️⃣  Building AI agent with RAG memory..."
	uv run src/cli.py build-agent --agent-type llm --memory-size 1000

compile-workflow:
	@echo "Step 7️⃣  Compiling NemoIR workflow..."
	uv run src/cli.py compile-workflow --workflow src/workflows/agent_workflow.nemo \
	                                  --output compiled_agent.ir --optimize aggressive

benchmark:
	@echo "Step 8️⃣  Running comprehensive benchmarks..."
	uv run src/cli.py benchmark --baselines all --num-scenarios 100 --output experiments/results/

online-demo:
	@echo "Step 9️⃣  Running online execution demo (10 seconds)..."
	uv run src/cli.py run-online --agent compiled --duration 10 --log logs/online_demo.log

launch-autoresearch:
	@echo "Step 🔟  Launching AutoResearch loop..."
	@echo "   This will run autonomously for 8 hours (or specify duration)"
	uv run src/cli.py launch-autoresearch --duration 28800 --gpu 0

# Development & Testing
test:
	@echo "🧪 Running test suite..."
	pytest tests/ -v --cov=src --cov-report=html
	@echo "✅ Tests complete! Coverage report: htmlcov/index.html"

lint:
	@echo "🔍 Running linters..."
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports
	@echo "✅ Linting complete!"

format:
	@echo "✨ Formatting code..."
	black src/ tests/
	ruff check src/ tests/ --fix
	@echo "✅ Formatting complete!"

# Documentation
docs:
	@echo "📚 Building documentation..."
	@echo "   See docs/ directory:"
	@echo "     - README.md (project overview)"
	@echo "     - ARCHITECTURE.md (system design)"
	@echo "     - PIPELINE.md (detailed 10-step walkthrough)"
	@echo "     - RESEARCH_PROTOCOL.md (experiment methodology)"
	@echo "     - API_REFERENCE.md (agent tool API)"
	@echo "     - DEPLOYMENT.md (online execution setup)"
	@echo "✅ Documentation ready!"

# Cleanup
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov
	rm -f .coverage
	rm -rf build dist *.egg-info
	rm -f logs/*.log
	@echo "✅ Cleanup complete!"

clean-data:
	@echo "🗑️  Removing generated datasets (WARNING: irreversible)..."
	rm -rf data/raw data/processed data/ground_truth data/models data/indices
	rm -rf experiments/baseline_models experiments/adversarial_water_filling experiments/diffract_optimization
	rm -rf experiments/rl_agent experiments/llm_agent experiments/nemoir_compiled experiments/autoresearch_logs
	rm -rf research/experiment_logs
	@echo "✅ Data cleanup complete!"

# Advanced targets
train-rl-advanced:
	@echo "Advanced: Training RL agent with distributed PPO..."
	uv run src/models/reinforcement_learning.py --algorithm ppo_distributed --workers 4 --gpu 0

train-supervised-transfer:
	@echo "Advanced: Transfer learning from pretrained models..."
	uv run src/models/supervised_learning.py --pretrained gpt2 --freeze-backbone

optimize-nmf-factorization:
	@echo "Advanced: Optimize NMF factorization for power allocation..."
	uv run src/optimization/adversarial_water_filling.py --solver nmf --iterations 1000

visualize-agent-graph:
	@echo "Advanced: Visualize compiled agent workflow graph..."
	uv run src/workflows/visualize_workflow.py --input compiled_agent.ir --output agent_graph.html

# CI/CD helpers
ci-test: lint test
	@echo "✅ CI checks passed!"

# Development server (for API/interactive use)
serve:
	@echo "🌐 Starting development server on http://localhost:8000"
	uv run uvicorn src.agent.api:app --reload --port 8000

# Jupyter notebook
notebook:
	@echo "📓 Starting Jupyter Lab..."
	uv run jupyter lab notebooks/

# Continuous monitoring
watch-results:
	@echo "👁️  Watching experiment results (updates every 5s)..."
	watch -n 5 'cat research/results.tsv | tail -20'

watch-logs:
	@echo "👁️  Watching autoresearch logs..."
	tail -f research/experiment_logs/*.log

# Default: show help
.DEFAULT_GOAL := help
