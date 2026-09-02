# AI Spectrum Management for Non-terrestrial Networks (NTN)
## By AI Agentic Resource Allocation

> An end-to-end research framework for dynamic spectrum management in 6G wireless networks using AI agents, adversarial game theory, and autonomous optimization loops.

## Overview

This project integrates **three core research frameworks** to address spectrum efficiency and resource allocation in adversarial non-terrestrial network (NTN) environments:

- **RAG-Practice**: Multi-modal retrieval-augmented generation for LLM-based agent reasoning and memory
- **AutoResearch**: Autonomous AI-driven experiment loop for continuous algorithm improvement
- **NemoIR**: Compiler framework for optimizing agent workflows into low-latency executables

### The Research Problem

In 6G wireless networks—especially LEO satellite and dense terrestrial deployments—spectrum is contested, interference is adversarial, and traditional iterative optimization (water-filling) produces suboptimal Nash equilibria. This project develops:

1. **Adversarial Water-Filling**: A minimax formulation for competitive power allocation under intentional jamming
2. **AI-Agentic Reasoning**: LLM-based agents that autonomously diagnose network conditions and select optimization strategies
3. **Real-time Execution**: NemoIR-compiled workflows for millisecond-scale spectrum decisions
4. **Bounded Model Selection**: AutoResearch runs reproducible hyperparameter trials, evaluates held-out model quality, and records keep/discard evidence

---

## 10-Step Integrated Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ 1. DATASET BUILDING                                                 │
│    ↓ Scenario/Topology → Channel → Traffic → Interference → Jammer  │
│    Output: Raw realistic NTN wireless environments                   │
├─────────────────────────────────────────────────────────────────────┤
│ 2. DATA PROCESSING & STATE REPRESENTATION                           │
│    ↓ Cleaning → Feature Engineering → State Construction            │
│    Output: Standardized learning states, train/val/test split        │
├─────────────────────────────────────────────────────────────────────┤
│ 3. GROUND-TRUTH OPTIMIZATION                                        │
│    ↓ Adversarial Water-Filling | DIFFRACT → Teacher Dataset        │
│    Output: High-quality optimal/near-optimal solutions              │
├─────────────────────────────────────────────────────────────────────┤
│ 4. NEURAL MODEL TRAINING                                            │
│    ↓ Supervised/RL/Prediction Models trained on teacher dataset     │
│    Output: Fast function approximators & predictors                 │
├─────────────────────────────────────────────────────────────────────┤
│ 5. AI AGENT CONSTRUCTION                                            │
│    ↓ Reasoning + Planning + Tool Selection + Memory (RAG)           │
│    Output: Autonomous agent for network diagnosis & optimization    │
├─────────────────────────────────────────────────────────────────────┤
│ 6. AGENT TOOL LAYER                                                 │
│    ↓ estimate_channel() | detect_jamming() | allocate_power() etc.  │
│    Output: Reliable wireless network functions for agent calls      │
├─────────────────────────────────────────────────────────────────────┤
│ 7. AGENT WORKFLOW & NemoIR COMPILATION                              │
│    ↓ Agent Graph → Workflow IR → GPU-Scheduled Execution            │
│    Output: Optimized, low-latency executable workflow               │
├─────────────────────────────────────────────────────────────────────┤
│ 8. EVALUATION & BENCHMARKING                                        │
│    ↓ Spectral Eff. | Latency | Fairness | Jamming Robustness       │
│    Output: Performance metrics, comparison matrix                   │
├─────────────────────────────────────────────────────────────────────┤
│ 9. ONLINE NTN EXECUTION                                             │
│    ↓ Real-time Observation → Reasoning → Optimization → Reconfig    │
│    Output: Live spectrum allocation & adaptive beamforming          │
├─────────────────────────────────────────────────────────────────────┤
│ 10. AUTORESEARCH LOOP (Autonomous Improvement)                      │
│     ↓ Train Trial → Held-Out Evaluation → Keep/Reject → Log Result  │
│     Output: Auditable model-selection evidence                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Architecture Map

The tree below shows the intended project layout. The implemented runtime modules are the files used by the commands in the sections that follow; some fine-grained files shown here remain planned extensions.

```
ai-spectrum-ntn/
├── docs/                               # Research documentation
│   ├── ARCHITECTURE.md                 # System design overview
│   ├── PIPELINE.md                     # Detailed pipeline walkthrough
│   └── INTEGRATION_GUIDE.md            # Framework integration guide
│
├── src/
│   ├── dataset_generation/             # Step 1: Generate raw data
│   │   ├── __init__.py
│   │   ├── scenario_topology.py        # NTN scenario & satellite topology
│   │   ├── channel_generation.py       # Wireless channel models
│   │   ├── traffic_generation.py       # User traffic & demand
│   │   ├── interference_modeling.py    # Multi-user interference
│   │   ├── adversarial_jammer.py       # Adversarial jamming generation
│   │   └── dataset_assembler.py        # Raw dataset creation
│   │
│   ├── data_processing/                # Step 2: Process to learning states
│   │   ├── __init__.py
│   │   ├── cleaning.py                 # Data validation & cleaning
│   │   ├── feature_engineering.py      # Feature extraction & encoding
│   │   ├── state_construction.py       # Build learning state vectors
│   │   ├── data_splitting.py           # Train/val/test partitioning
│   │   └── preprocessing_pipeline.py   # Orchestrate data pipeline
│   │
│   ├── optimization/                   # Step 3: Ground-truth solutions
│   │   ├── __init__.py
│   │   ├── water_filling.py            # Classical water-filling algorithm
│   │   ├── adversarial_water_filling.py # Minimax game-theoretic formulation
│   │   ├── diffract_optimizer.py       # DIFFRACT optimization solver
│   │   ├── constraint_handling.py      # Power/interference/spectrum constraints
│   │   └── teacher_dataset.py          # Generate training labels
│   │
│   ├── models/                         # Step 4: Neural model training
│   │   ├── __init__.py
│   │   ├── supervised_learning.py      # Imitation learning from teacher
│   │   ├── reinforcement_learning.py   # RL agent for dynamic allocation
│   │   ├── channel_predictor.py        # Predict channel state evolution
│   │   ├── traffic_predictor.py        # Forecast traffic demand
│   │   ├── jammer_detector.py          # Detect adversarial jamming
│   │   ├── uncertainty_estimator.py    # Bayesian uncertainty quantification
│   │   └── model_zoo.py                # Pre-trained model registry
│   │
│   ├── agent/                          # Step 5: AI Agent construction
│   │   ├── __init__.py
│   │   ├── ai_agent.py                 # Main agent orchestrator
│   │   ├── environment_perception.py   # Observe network state
│   │   ├── state_understanding.py      # Interpret observations
│   │   ├── reasoning.py                # LLM-based reasoning & planning
│   │   ├── memory_rag.py               # RAG-based memory system
│   │   ├── action_generator.py         # Generate optimization actions
│   │   └── agent_config.yaml           # Agent hyperparameters
│   │
│   ├── tools/                          # Step 6: Wireless tools for agents
│   │   ├── __init__.py
│   │   ├── wireless_tools.py           # Core wireless functions (Sionna)
│   │   ├── channel_estimator.py        # estimate_channel()
│   │   ├── jammer_detector_tool.py     # detect_jamming()
│   │   ├── traffic_forecaster.py       # predict_traffic()
│   │   ├── power_allocator.py          # allocate_power()
│   │   ├── spectrum_allocator.py       # allocate_spectrum()
│   │   ├── beam_manager.py             # reconfigure_beam()
│   │   ├── handover_manager.py         # handover_user()
│   │   ├── policy_evaluator.py         # evaluate_policy()
│   │   └── tool_registry.py            # Register agent tools
│   │
│   ├── workflows/                      # Step 7: NemoIR workflow compilation
│   │   ├── agent_workflow.nemo         # Agent FSM in NemoIR DSL
│   │   ├── spectrum_mgmt_workflow.nemo # Spectrum management workflow
│   │   ├── emergency_response.nemo     # Jamming response workflow
│   │   └── workflows.md                # Workflow documentation
│   │
│   └── evaluation/                     # Steps 8-9: Benchmarking & execution
│       ├── __init__.py
│       ├── benchmarks.py               # Run all baseline comparisons
│       ├── metrics.py                  # Compute spectral efficiency, latency, etc.
│       ├── baseline_waterfilling.py    # Classical water-filling reference
│       ├── comparison.py               # Side-by-side metric comparison
│       ├── online_executor.py          # Real-time execution harness
│       └── results_aggregator.py       # Collect & visualize results
│
├── experiments/                        # Experiment outputs & logs
│   ├── baseline_models/                # Classical optimization baselines
│   ├── adversarial_water_filling/      # AWF approach results
│   ├── diffract_optimization/          # DIFFRACT solver results
│   ├── rl_agent/                       # RL agent training & evaluation
│   ├── llm_agent/                      # LLM agent results
│   ├── nemoir_compiled/                # NemoIR-compiled workflow perf
│   └── autoresearch_logs/              # Autonomous research loop logs
│
├── research/                           # Step 10: Autonomous research loop
│   ├── program.md                      # Agent research instructions
│   ├── autoresearch_config.py          # Autoresearch experiment config
│   ├── experiment_runner.py            # Launch autonomous experiments
│   ├── results.tsv                     # Experiment results tracking
│   ├── learnings.md                    # Recorded research insights
│   └── experiment_logs/                # Individual experiment outputs
│
├── data/                               # Data artifacts
│   ├── raw/                            # Raw simulated datasets
│   ├── processed/                      # Processed learning states
│   ├── ground_truth/                   # Teacher labels from optimization
│   ├── models/                         # Trained model checkpoints
│   └── indices/                        # FAISS indices for RAG
│
├── notebooks/                          # Jupyter analysis & exploration
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_optimization_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_agent_inference.ipynb
│   ├── 05_benchmark_results.ipynb
│   └── 06_research_insights.ipynb
│
├── config/                             # Configuration files
│   ├── default_config.yaml             # System-wide defaults
│   ├── ntn_scenario.yaml               # NTN topology & parameters
│   ├── training_config.yaml            # Model training hyperparameters
│   ├── agent_config.yaml               # Agent behavior parameters
│   └── evaluation_config.yaml          # Benchmark parameters
│
├── pyproject.toml                      # Unified Python project config
├── uv.lock                             # Locked dependencies
├── Makefile                            # Common tasks
└── LICENSE
```

---

## Installation

### Prerequisites
- **Python 3.10+**
- **NVIDIA GPU** (H100 recommended, tested on V100+)
- **uv** package manager ([install](https://docs.astral.sh/uv/))

### Quick Start

```bash
# 1. Navigate to project root
cd ai-spectrum-ntn

# 2. Install dependencies (all three frameworks integrated)
uv sync

# 3. Generate a small raw NTN dataset
uv run src/cli.py generate-dataset --num-scenarios 10 --output data/raw

# 4. Build learning states and teacher labels
uv run src/cli.py process-data --input data/raw --output data/processed
uv run src/cli.py generate-labels --method adversarial_wf --num-scenarios 10

# 5. Train a local smoke-test policy
uv run src/cli.py train-model --model supervised --epochs 2 --device cpu
```

---

## Key Concepts

### 1. **Adversarial Water-Filling (AWF)**
Formulates spectrum allocation as a minimax game:
```
maximize   min       f(power, channel, jammer)
  p      jammer
```
Solutions are more robust to adversarial interference than classical water-filling.

### 2. **AI Agentic Orchestration**
The agent:
- **Perceives** network state (channels, traffic, jamming)
- **Reasons** about conditions using LLM + domain knowledge
- **Plans** which optimization strategy to invoke
- **Executes** via tool calls (DIFFRACT, AWF, RL predictor)
- **Learns** from outcomes and failure cases (AutoResearch loop)

### 3. **NemoIR Workflow Compilation**
NemoIR compiles the declared agent workflow into validated Workflow IR. The project includes a visualizer artifact at `data/models/agent_workflow.html`; runtime latency claims require separate target-runtime benchmarking.

### 4. **AutoResearch Model Selection**
Runs bounded, reproducible supervised-policy trials:
- Trains each candidate on the chronological training split
- Evaluates test MSE, spectral efficiency, and inference latency on held-out states
- Selects by lowest test MSE, using spectral efficiency as a tie-breaker
- Logs all results without modifying source code

---

## Quick Examples

### Train the Supervised Model
```bash
uv run src/cli.py train-model --model supervised --epochs 100 --device cpu
```

### Run GPU AutoResearch Model Trials
```bash
sbatch scripts/autoresearch.sbatch
```

### Benchmark Allocation Policies
```bash
uv run src/cli.py benchmark --baselines all --num-scenarios 10
```

### Visualize Agent Workflow (NemoIR)
```bash
$HOME/.local/bin/nemo compile src/workflows/agent_workflow.nemo \
  --target visualizer --output data/models/agent_workflow.html
```

---

## Integration of Three Frameworks

| Framework | Role | Integration Point |
|-----------|------|-------------------|
| **RAG-Practice** | Memory + LLM Reasoning | `src/agent/memory_rag.py` — stores/retrieves domain knowledge for agent reasoning |
| **AutoResearch** | Autonomous Algorithm Improvement | `research/` folder — runs experiment loop to improve models & algorithms |
| **NemoIR** | Workflow Compiler | `src/workflows/` — compiles agent workflows to optimized executables |

---

## Running Your First Experiment

### Experiment 1: Generate and Process Data
```bash
uv run src/cli.py generate-dataset --num-scenarios 10 --output data/raw
uv run src/cli.py process-data --input data/raw --output data/processed
```

### Experiment 2: Generate Teacher Labels and Benchmark
```bash
uv run src/cli.py generate-labels --method adversarial_wf --num-scenarios 10
uv run src/cli.py benchmark --baselines all --num-scenarios 10
```

### Experiment 3: Train and Evaluate Model Candidates on Slurm
```bash
sbatch scripts/autoresearch.sbatch
```

---

## Contribution & Research Protocol

1. **Read**: [PIPELINE.md](docs/PIPELINE.md)
2. **Implement**: Add your algorithm to `src/optimization/` or `src/models/`
3. **Evaluate**: Use `src/evaluation/benchmarks.py` for fair comparison
4. **Log**: Review `research/model_results_v2.tsv` and `research/model_learnings_v2.md`
5. **Iterate**: Expand the dataset before treating model-selection results as research conclusions

---

## Performance Targets

| Metric | Target | Baseline |
|--------|--------|----------|
| Spectral Efficiency (bits/Hz/s) | 6.5+ | 5.2 (classical WF) |
| Latency (decision, ms) | <10 | 100+ (iterative) |
| Jamming Robustness | >90% | 60% (classical) |
| Training Time (per epoch) | <2 min | 8 min |

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@inproceedings{spectrum-ntn-ai-agents,
  title={AI Agentic Spectrum Management for NTN: Adversarial Optimization meets Autonomous Workflows},
  author={Your Name},
  booktitle={Proceedings of 6G Research Workshop},
  year={2025}
}
```

---

## License

MIT License — See [LICENSE](LICENSE) file.

---

## Contact & Support

- **Issues**: GitHub Issues
- **Research Questions**: Create a Discussion
- **Collaboration**: Email or submit PR with research proposal

---

**Status**: Research prototype. All ten pipeline stages have CPU smoke coverage; GPU training runs are submitted through Slurm.
