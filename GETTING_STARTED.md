# Getting Started: Initialization & First Steps

This document guides you through initializing the project and running your first experiments.

## Prerequisites Checklist

Before starting, ensure you have:

- [ ] Python 3.10+ installed (`python --version`)
- [ ] NVIDIA GPU available (`nvidia-smi`)
- [ ] `uv` package manager installed (`uv --version`)
- [ ] Git repository initialized (`git status`)
- [ ] CUDA 12.1+ (for PyTorch)
- [ ] At least 50GB free disk space (for datasets)
- [ ] Environment variables set:
  ```bash
  export OPENAI_API_KEY=sk-...      # For OpenAI LLM
  export ANTHROPIC_API_KEY=sk-...   # For Claude (optional)
  export CUDA_VISIBLE_DEVICES=0     # GPU device ID
  ```

## Phase 1: Installation (5 minutes)

### Step 1A: Install Dependencies
```bash
cd /home/msai/qinh0007/ai-spectrum-ntn
make install
```

**What it does**:
- Runs `uv sync` to install all 70+ dependencies
- Creates virtual environment
- Compiles PyTorch CUDA extensions

**Expected output**:
```
📦 Installing dependencies...
[installing 70+ packages...]
✅ Installation complete!
```

**If errors occur**:
- Check CUDA version: `nvcc --version` (must be 12.1+)
- Check disk space: `df -h` (need 30GB+ for torch)
- Try: `pip install --upgrade setuptools wheel`

### Step 1B: Verify Installation
```bash
# Check PyTorch CUDA availability
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check LangChain + RAG imports
python -c "from langchain.chat_models import ChatOpenAI; print('LangChain: OK')"

# Check Sionna (wireless)
python -c "import sionna; print(f'Sionna: {sionna.__version__}')"
```

## Phase 2: Project Setup (5 minutes)

### Step 2A: Create Directory Structure
```bash
make setup
```

**What it does**:
- Creates data directories: `data/raw/`, `data/processed/`, `data/ground_truth/`, `data/models/`, `data/indices/`
- Creates experiments directories for each method
- Creates research directories for AutoResearch
- Creates logs and notebooks directories

**Expected structure**:
```
ai-spectrum-ntn/
├── data/
│   ├── raw/                    ← Generated raw datasets
│   ├── processed/              ← Processed learning states
│   ├── ground_truth/           ← Optimal labels
│   ├── models/                 ← Trained model checkpoints
│   └── indices/                ← FAISS vector indices
├── experiments/
│   ├── baseline_models/
│   ├── adversarial_water_filling/
│   ├── rl_agent/
│   └── ...
├── research/
│   ├── experiment_logs/
│   ├── results.tsv
│   └── learnings.md
└── logs/                       ← Pipeline execution logs
```

### Step 2B: Verify Setup
```bash
# List created directories
ls -la data/ experiments/ research/

# Check configuration
cat config/default_config.yaml | head -20
```

## Phase 3: Quick Start (2 hours)

### Option A: Minimal Demo (10 scenarios, ~30 minutes)
```bash
make quickstart
```

**What it does**:
1. Generates 10 NTN scenarios
2. Processes data to learning states
3. Generates ground-truth optimal allocations (adversarial water-filling)
4. Trains supervised imitation learning model (10 epochs)

**Expected output**:
```
Step 1️⃣  Generating NTN dataset...
Generated 10 scenarios ✅

Step 2️⃣  Processing data to learning states...
Created learning states ✅

Step 3️⃣  Generating ground-truth optimal labels...
Generated 10 optimal allocations ✅

Step 4️⃣  Training supervised model...
Epoch 1/10: loss=0.523 ✅
...
Epoch 10/10: loss=0.087 ✅

✅ Quick-start complete!
```

**If errors occur**:
- Check CUDA memory: `nvidia-smi` (need ~4GB for training)
- Reduce batch size in config/default_config.yaml: `batch_size: 16` (from 32)
- Check logs: `tail -f logs/pipeline.log`

### Option B: Full Pipeline Demo (100 scenarios, ~2 hours)
```bash
make pipeline-demo
```

**What it does**:
- Runs Steps 1-8 with 100 scenarios
- Generates datasets, processes, optimizes, trains, benchmarks

## Phase 4: Explore Results

### View Dataset Exploration
```bash
# Check raw dataset
ls -lh data/raw/
ls -lh data/raw/scenarios_*.json | head -5

# Check processed data
ls -lh data/processed/
python -c "import numpy as np; data = np.load('data/processed/learning_states.npz'); print(data.files)"
```

### View Training Results
```bash
# Check trained models
ls -lh data/models/
python -c "import torch; model = torch.load('data/models/supervised_latest.pt'); print(model)"

# Monitor training logs
tail -f logs/pipeline.log
```

### View Benchmark Results
```bash
# Open results
cat experiments/results/benchmark_summary.txt

# Plot comparison
python -c "import matplotlib.pyplot as plt; data = np.load('experiments/results/comparison_metrics.npz'); print(data.files)"
```

## Phase 5: Run Individual Pipeline Steps

Once quick-start works, you can run individual steps:

### Generate Larger Dataset
```bash
# 1,000 scenarios for real training
uv run src/cli.py generate-dataset --num-scenarios 1000 --output data/raw/

# Track progress
tail -f logs/dataset_generation.log
```

### Train Individual Models
```bash
# Supervised learning (imitation)
uv run src/cli.py train-model --model supervised --epochs 100 --gpu 0

# RL agent (PPO)
uv run src/cli.py train-model --model rl --epochs 1000 --gpu 0

# Channel predictor (LSTM)
uv run src/cli.py train-model --model channel_predictor --epochs 50 --gpu 0
```

### Run Benchmarking
```bash
# Compare all baselines
uv run src/cli.py benchmark --baselines all --num-scenarios 100 --output experiments/results/

# View results
cat experiments/results/benchmark_summary.txt
python experiments/results/plot_comparison.py
```

### Build and Test Agent
```bash
# Build AI agent with RAG memory
uv run src/cli.py build-agent --agent-type llm --memory-size 1000

# Test agent on sample scenario
python -c "from src.agent.ai_agent import AIAgent; agent = AIAgent(agent_type='llm'); cmd = agent.allocate_spectrum([...state...]); print(cmd)"
```

## Phase 6: Integration Testing

### Test RAG-Practice Integration
```bash
# Create RAG index with wireless knowledge
python -c "
from src.agent.memory_rag import MemoryRAG
docs = [
    'Adversarial water-filling minimizes worst-case jamming throughput',
    'CSI quantization reduces feedback overhead',
    'Beam training increases spectral efficiency'
]
rag = MemoryRAG()
rag.add_documents(docs)
results = rag.retrieve('jamming attack')
print(results)
"

# Test LLM reasoning
uv run src/agent/reasoning.py --test-scenario jamming_detected
```

### Test AutoResearch Integration
```bash
# Initialize experiment tracking
uv run research/experiment_runner.py --init

# Run a single experiment with new hypothesis
uv run research/experiment_runner.py \
  --hypothesis "use_transformer_for_channel_prediction" \
  --gpu 0

# View results
cat research/results.tsv | tail -10
cat research/learnings.md | tail -20
```

### Test NemoIR Integration
```bash
# Download/install NemoIR (if not already installed)
pip install nemoir  # or: cargo install nemoir

# Compile workflow to IR
nemoir compile src/workflows/agent_workflow.nemo \
  --output-ir agent.ir \
  --optimize aggressive \
  --target gpu

# Visualize compiled workflow
nemoir visualize agent.ir --output agent_graph.html
open agent_graph.html  # View in browser
```

## Phase 7: Development Setup

### Optional: Install Development Tools
```bash
# Code formatting & linting
uv run black --version && ruff --version && mypy --version

# Testing framework
uv run pytest --version

# Jupyter for interactive analysis
uv run jupyter --version
```

### Create First Jupyter Notebook
```bash
# Start Jupyter Lab
make notebook

# Or create new notebook
jupyter notebook notebooks/01_dataset_exploration.ipynb
```

### Example notebook cells:
```python
# Cell 1: Load and explore dataset
import numpy as np
data = np.load('data/processed/learning_states.npz')
print(f"States shape: {data['states'].shape}")
print(f"Actions shape: {data['actions'].shape}")

# Cell 2: Plot spectral efficiency distribution
import matplotlib.pyplot as plt
plt.hist(data['labels'][:, 0], bins=50)  # Spectral efficiency
plt.xlabel('Spectral Efficiency (bits/Hz)')
plt.ylabel('Frequency')
plt.title('Dataset Distribution')
plt.show()

# Cell 3: Check for class imbalance
print(f"Mean spectral efficiency: {data['labels'][:, 0].mean():.2f}")
print(f"Std spectral efficiency: {data['labels'][:, 0].std():.2f}")
```

## Phase 8: Continuous Monitoring

### Watch Experiment Results
```bash
# Monitor results in real-time
make watch-results

# Monitor logs in real-time
make watch-logs

# Or use custom monitoring
watch -n 5 'tail -20 research/results.tsv'
```

## Troubleshooting Guide

### Issue: CUDA Out of Memory
**Solution**:
```bash
# Reduce batch size
sed -i 's/batch_size: 32/batch_size: 16/' config/default_config.yaml

# Or limit GPU memory
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512
```

### Issue: Import Errors
**Solution**:
```bash
# Reinstall in development mode
uv pip install -e .

# Update dependencies
uv sync --upgrade
```

### Issue: Dataset Generation Hangs
**Solution**:
```bash
# Check available disk space
df -h data/

# Increase GPU memory allocation
nvidia-smi -pm 1 -i 0  # Persistent mode

# Reduce scenarios per batch
uv run src/cli.py generate-dataset --num-scenarios 100 --batch-size 10
```

### Issue: LLM API Errors
**Solution**:
```bash
# Check API keys
echo $OPENAI_API_KEY | cut -c1-20
echo $ANTHROPIC_API_KEY | cut -c1-20

# Test LLM connection
python -c "from langchain.chat_models import ChatOpenAI; llm = ChatOpenAI(model='gpt-4'); print(llm.invoke('Hello'))"
```

## Next: Implementation Guide

After completing this initialization, you're ready to:

1. **Implement Step 1**: See [PIPELINE.md - Step 1](docs/PIPELINE.md#step-1-dataset-generation)
2. **Understand Architecture**: See [ARCHITECTURE.md](docs/ARCHITECTURE.md)
3. **Integrate Frameworks**: See [INTEGRATION_GUIDE.md](docs/INTEGRATION_GUIDE.md)

## Support Resources

- **Project Documentation**: `docs/` directory
- **Configuration Reference**: `config/default_config.yaml`
- **CLI Commands**: `make help` or `uv run src/cli.py --help`
- **Make Targets**: `make help`
- **Logs**: `logs/` directory
- **Experiment Results**: `experiments/results/` and `research/`

## Estimated Timeline

| Phase | Tasks | Duration | Status |
|-------|-------|----------|--------|
| Installation | `make install` | 5 min | Ready |
| Setup | `make setup` | 2 min | Ready |
| Quick Start (10 scen) | Step 1-4 demo | 30 min | Ready |
| Full Pipeline (100 scen) | Step 1-8 demo | 2-3 hours | Ready |
| Integration Testing | RAG, AutoResearch, NemoIR | 1-2 hours | Ready |
| Full System (1000 scen) | All 10 steps | 1-2 days | Pending implementation |
| Autonomous Research | Step 10 loop | 8-24 hours | Pending implementation |

---

**Ready to start? Run this now:**
```bash
cd /home/msai/qinh0007/ai-spectrum-ntn
make install && make setup && make quickstart
```

**Then check results:**
```bash
ls -la data/
tail -100 logs/pipeline.log
cat experiments/results/benchmark_summary.txt
```

Enjoy! 🚀
