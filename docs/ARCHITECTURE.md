# System Architecture

## High-Level Design Philosophy

This project integrates **three complementary research frameworks** to solve the NTN spectrum management problem:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                     AI-AGENTIC SPECTRUM MANAGEMENT SYSTEM                 │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ╔═════════════════════════╗  ╔═════════════════════╗  ╔════════════════╗ │
│  ║   RAG-Practice          ║  ║   AutoResearch      ║  ║    NemoIR      ║ │
│  ║   (Agent Memory/LLM)    ║  ║  (Autonomous Loop)  ║  ║  (Compilation) ║ │
│  ╠═════════════════════════╣  ╠═════════════════════╣  ╠════════════════╣ │
│  ║ • LLM Reasoning         ║  ║ • Experiment Loop   ║  ║ • Workflow DSL ║ │
│  ║ • Knowledge Retrieval   ║  ║ • Hyperparameter    ║  ║ • Optimization ║ │
│  ║ • Tool Calling          ║  ║   Tuning            ║  ║ • Graph Compile║ │
│  ║ • Planning & Context    ║  ║ • Dataset Expansion ║  ║ • Parallel Exec║ │
│  ║ • Memory Management     ║  ║ • Automated Testing ║  ║ • GPU Schedule ║ │
│  ╚═════════════════════════╝  ╚═════════════════════╝  ╚════════════════╝ │
│           ▲                            ▲                         ▲         │
│           │                            │                         │         │
│           └────────────────────────────┼─────────────────────────┘         │
│                                        │                                   │
│                            ┌───────────────────────┐                       │
│                            │   AI Agent Core       │                       │
│                            │ (Reasoning + Planning)│                       │
│                            └───────────────────────┘                       │
│                                        ▲                                   │
│                      ┌─────────────────┼─────────────────┐                 │
│                      │                 │                 │                 │
│           ┌──────────▼────────┐  ┌────▼──────────┐  ┌───▼────────────┐    │
│           │  Perception       │  │  Models       │  │  Tools         │    │
│           │  (Observe State)  │  │  (Neural Nets)│  │  (Wireless Ops)│    │
│           └───────────────────┘  └───────────────┘  └────────────────┘    │
│                      ▲                 ▲                      ▲             │
│                      │                 │                      │             │
│           ┌──────────┴─────────────────┴──────────────────────┘             │
│           │                                                                │
│           ▼                                                                │
│  ┌────────────────────────────────────────────────────────┐                │
│  │     Optimization & Learning Pipeline                   │                │
│  │  (Ground-Truth Solutions → Supervised/RL Training)     │                │
│  └────────────────────────────────────────────────────────┘                │
│           ▲                                                                │
│           │                                                                │
│  ┌────────┴────────────────────────────────────────────────┐               │
│  │     Data Pipeline                                       │               │
│  │  (Raw Data → Cleaning → Features → Learning States)    │               │
│  └─────────────────────────────────────────────────────────┘              │
│           ▲                                                                │
│           │                                                                │
│  ┌────────┴────────────────────────────────────────────────┐               │
│  │     Dataset Generation                                  │               │
│  │  (Scenarios, Channels, Traffic, Interference, Jammers)  │               │
│  └─────────────────────────────────────────────────────────┘              │
│                                                                             │
└───────────────────────────────────────────────────────────────────────────┘
```

## Core Layers

### Layer 1: Dataset Generation
**Purpose**: Create realistic non-terrestrial network (NTN) scenarios

- **Scenario Topology**: LEO satellite constellation, user ground stations, interference patterns
- **Channel Models**: Pathloss, fading, Doppler effects for satellite-ground links
- **Traffic Models**: User demand, packet arrivals, QoS requirements
- **Interference**: Multi-user interference, adjacent-channel leakage, cross-links
- **Adversarial Jamming**: Strategic jammer placement, power levels, waveforms

**Key Module**: `src/dataset_generation/`

### Layer 2: Data Processing & Feature Engineering
**Purpose**: Convert raw simulation outputs → standardized learning states

- **Data Validation**: Check for outliers, missing values, inconsistencies
- **Feature Extraction**: Channel quality, interference power, traffic load, jammer signatures
- **State Encoding**: Vectorize topology, channel state, traffic demand, jammer activity
- **Normalization**: Standardize features for neural network training
- **Data Splitting**: Train/validation/test with temporal consistency

**Key Module**: `src/data_processing/`

### Layer 3: Ground-Truth Optimization
**Purpose**: Generate optimal/near-optimal solutions as training targets

- **Classical Water-Filling**: Watterson's algorithm for single-user scenario
- **Adversarial Water-Filling**: Minimax formulation for competitive allocation
- **DIFFRACT Solver**: Gradient-based optimization for constrained allocation
- **Teacher Dataset**: Collect (state, optimal_action) pairs for imitation learning

**Key Module**: `src/optimization/`

### Layer 4: Neural Model Training
**Purpose**: Learn fast approximations of expensive optimization

- **Supervised Learning**: Imitation learning from teacher solutions
- **Reinforcement Learning**: Policy gradient methods (PPO, A3C)
- **Predictive Models**: Channel state prediction, traffic forecasting, jammer detection
- **Uncertainty Quantification**: Bayesian uncertainty for risk-aware decisions

**Key Module**: `src/models/`

### Layer 5: AI Agent Construction
**Purpose**: Build autonomous reasoning agent

- **Perception**: Observe current network state, traffic, interference
- **State Understanding**: Interpret observations using embeddings
- **LLM Reasoning** (RAG-Practice): Diagnose conditions, select optimization strategy
- **Planning**: Decompose complex decisions into tool calls
- **Memory** (RAG-Practice): Store and retrieve domain knowledge
- **Action Generation**: Translate reasoning to concrete spectrum allocation commands

**Key Module**: `src/agent/`
**Integrated Framework**: RAG-Practice (LLM + memory)

### Layer 6: Agent Tool Layer
**Purpose**: Connect agent reasoning to reliable wireless operations

Each tool is a deterministic wireless function:

| Tool | Input | Output | Purpose |
|------|-------|--------|---------|
| `estimate_channel()` | Signal + Noise | Channel CSI | Estimate channel state |
| `detect_jamming()` | Received Signal | Jammer Profile | Identify adversarial interference |
| `predict_traffic()` | History | Forecast | Predict user demand |
| `allocate_power()` | Budget, Channel | Power Vector | Water-filling allocation |
| `allocate_spectrum()` | Demand, Interference | Spectrum Bands | Frequency assignment |
| `reconfigure_beam()` | Target, Channel | Beamweights | Update beam steering |
| `handover_user()` | User, Satellite Options | New Satellite | Switch connectivity |
| `evaluate_policy()` | State, Action | Reward | Compute performance metric |

**Key Module**: `src/tools/`

### Layer 7: Workflow Compilation (NemoIR)
**Purpose**: Optimize agent workflow for low-latency execution

- **Workflow DSL** (`.nemo`): Define agent as FSM + decision nodes
- **Compilation**: Convert to Intermediate Representation (IR)
- **Optimization**: Identify parallelizable tasks, GPU scheduling
- **Execution**: Deploy compiled workflow for real-time decisions
- **Speedup**: Reduce latency from 100ms (iterative) → 10ms (compiled)

**Key Module**: `src/workflows/`
**Integrated Framework**: NemoIR compiler

### Layer 8: Evaluation & Benchmarking
**Purpose**: Measure and compare approaches

**Metrics**:
- **Spectral Efficiency**: bits/Hz/s (throughput)
- **Latency**: Decision time, end-to-end delay
- **Fairness**: Coefficient of variation across users
- **Energy Efficiency**: bits/Joule
- **Robustness**: Performance degradation under jamming
- **Recovery Time**: Time to restore performance after interference

**Baselines**:
1. Classical Water-Filling (WF)
2. Adversarial Water-Filling (AWF)
3. DIFFRACT solver
4. RL agent (PPO)
5. LLM agent (GPT-based reasoning)
6. Proposed: Agent + DIFFRACT
7. Proposed: NemoIR-compiled agent

**Key Module**: `src/evaluation/`

### Layer 9: Online Execution
**Purpose**: Real-time operation in live NTN networks

- **Real-Time Observation**: Stream channel state, traffic, jamming
- **State Estimation**: Filter + predict next state
- **Agent Reasoning**: LLM diagnoses, selects tools
- **Optimization**: Run selected tool (WF/AWF/RL/DIFFRACT)
- **Reconfiguration**: Push allocation to satellite/ground equipment
- **Safety Fallbacks**: Revert to conservative allocation if confidence low

**Key Module**: `src/evaluation/online_executor.py`

### Layer 10: Autonomous Research Loop (AutoResearch)
**Purpose**: Continuously improve algorithms without human intervention

**The Loop**:
1. **Observe Failures**: Analyze when agent makes suboptimal decisions
2. **Hypothesis**: Generate new idea (modify model architecture, dataset, hyperparams)
3. **Experimentation**: Train on GPU with 5-minute fixed budget
4. **Evaluation**: Compute val metric (spectral efficiency, latency)
5. **Decision**: Keep if improved, discard otherwise
6. **Logging**: Record in `results.tsv` with description
7. **Knowledge Update**: Store learnings in `learnings.md`
8. **Loop**: Repeat

**Key Module**: `research/`
**Integrated Framework**: AutoResearch + Anthropic/Claude API for agent direction

---

## Data Flow & Component Interactions

### Scenario A: Training Phase (Offline)
```
Dataset Generation → Data Processing → Optimization (Teacher) → Model Training
       │                  │                     │                    │
   Sionna Sim         Feature Eng            CVXPY/DIFFRACT       PyTorch/TF
   (NTN envs)         State vectors         Optimal actions      Supervised/RL
       │                  │                     │                    │
       ▼                  ▼                     ▼                    ▼
   Raw data         Normalized states      (state, action) pairs    Models
   (~10GB)          (↓train/val/test)      (teacher dataset)     (trained)
```

### Scenario B: Inference Phase (Online)
```
Current Network State → Agent Perception → LLM Reasoning → Tool Selection
         │                     │                  │               │
  CSI, Traffic, Jamming    Embedding         Memory (RAG)    Allocate/Beam/HO
         │                     │                  │               │
         ▼                     ▼                  ▼              ▼
   Observation           State Vector        LLM Output     Wireless Actions
  (real-time)         (normalized)         (tool calls)    (spectrum changes)
                                               ▲                    │
                                               │                    │
                                     ┌─────────┴─────────┐          │
                                     │                   │          │
                                  Models              Tools         │
                                (Predictors)      (WF/DIFFRACT/     │
                                                    RL/Beams)       │
                                                                    ▼
                                                          Network Reconfig
                                                        (spectrum, power,
                                                         beams, handovers)
```

### Scenario C: AutoResearch Loop (Autonomous)
```
Offline Results → Failure Analysis → Hypothesis Generation → Code Modification
       │                │                    │                      │
   Experiment        Which scenarios      New: algorithm?       src/train.py
   outcomes          failed?              New: dataset?         or
                                          New: hyperparams?      src/models/
                                                                src/optimization/
       │◄──────────────────────────────────────────────────────────│
       │                                                             │
       ├─→ Git Commit → Training (5 min) → Evaluate → Log Results ──┘
       │                                                 │
       └─→ If improved: KEEP | Else: DISCARD ───→ Next Hypothesis
```

---

## Integration Points

### RAG-Practice Integration
- **LLM Reasoning**: `src/agent/reasoning.py` → Uses LangChain + OpenAI/Claude
- **Memory System**: `src/agent/memory_rag.py` → FAISS vector DB + embeddings
- **Tool Calling**: `src/agent/action_generator.py` → Structured tool invocation
- **Knowledge Base**: Stores domain facts about 6G, jamming, satellites

### AutoResearch Integration
- **Autonomous Loop**: `research/` folder → Runs experiments 24/7
- **Code Modification**: Agent edits `train.py`, `models/`, `optimization/`
- **Experiment Tracking**: `results.tsv` records all runs
- **Learnings Registry**: `research/learnings.md` captures insights

### NemoIR Integration
- **Workflow Definition**: `src/workflows/*.nemo` → NemoIR DSL
- **Compilation**: `nemoir compile agent_workflow.nemo` → Generate IR
- **Execution Backends**: Python, browser, distributed
- **Performance**: Parallel tool execution + GPU scheduling

---

## Key Design Decisions

### 1. Modular Pipeline
Each step (dataset → processing → optimization → models → agent → tools → workflows → evaluation) is **independently testable** and **swappable**.

### 2. Ground-Truth Optimization as Supervision
Rather than pure RL trial-and-error, we **generate optimal solutions** (teacher dataset) to guide learning. This achieves **faster convergence** and **guaranteed lower bounds** on performance.

### 3. LLM as Orchestrator (Not Solver)
The agent doesn't **compute** spectrum allocation (too slow), but **reasons** about conditions and **selects tools** (DIFFRACT, RL, WF). This gives **interpretability** + **reliability**.

### 4. Workflow Compilation for Real-Time
Converting agent FSM → compiled graph **parallelizes** decision-making and **GPU-schedules** numeric operations, reducing decision latency from **100ms → 10ms**.

### 5. AutoResearch for Continuous Improvement
Rather than static models, the system **autonomously experiments** overnight, finds better algorithms, and **updates** without human intervention.

---

## Scalability Considerations

| Component | Scale | Challenge | Solution |
|-----------|-------|-----------|----------|
| Dataset | 100k scenarios | Storage, I/O | Streaming + caching |
| Training | 10B+ parameters | VRAM, speed | Distributed, quantization |
| Inference | 1k users/satellite | Latency | NemoIR compilation, batching |
| Memory (RAG) | 1M+ facts | Retrieval speed | Hierarchical indexing, summarization |
| AutoResearch | 1000+ experiments | Compute hours | Distributed GPU cluster, efficient metrics |

---

## Next Steps for Implementation

1. ✅ **Architecture Design** (this document)
2. 🔄 **Implement Pipeline Modules** (datasets → models → agent)
3. 🔄 **Integrate RAG-Practice** (LLM reasoning + memory)
4. 🔄 **Integrate AutoResearch** (experiment loop)
5. 🔄 **Integrate NemoIR** (workflow compilation)
6. 🔄 **Benchmark & Evaluate** (compare baselines)
7. 🔄 **Online Deployment** (real-time spectrum allocation)
