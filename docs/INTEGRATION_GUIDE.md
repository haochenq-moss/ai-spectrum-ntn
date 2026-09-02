# Integration Guide: Unifying RAG-Practice, AutoResearch, and NemoIR

This document explains how to integrate the three existing research frameworks into the unified AI Spectrum Management project.

## Overview

The new project structure (`ai-spectrum-ntn/`) serves as the **main orchestrator** that:
1. Imports and wraps functionality from RAG-Practice, AutoResearch, and NemoIR
2. Creates specialized modules that implement the 10-step pipeline
3. Provides unified CLI and Makefile for end-to-end experiments
4. Maintains clear separation of concerns while enabling deep integration

```
ai-spectrum-ntn/ (NEW - orchestrator)
├── Imports from: RAG-Practice, AutoResearch, NemoIR
├── Uses: Sionna (wireless simulation), PyTorch/TF (training)
├── Implements: 10-step pipeline
└── Provides: CLI, API, Makefile

RAG-Practice/ (EXISTING - memory/reasoning)
├── LLM integration (OpenAI, Claude, HuggingFace)
├── RAG with FAISS vector DB
└── Tool-calling framework

AutoResearch/ (EXISTING - autonomous loop)
├── Experiment runner with 5-min budget
├── Autonomous hypothesis generation
├── Results tracking (results.tsv)
└── Git-based versioning

NemoIR/ (EXISTING - workflow compiler)
├── Workflow DSL (.nemo)
├── Compiler to IR
├── GPU scheduling
└── Multiple backend targets
```

---

## Step-by-Step Integration

### Phase 1: RAG-Practice Integration

**What to integrate**: LLM reasoning, memory management, tool-calling

**Where it fits in pipeline**:
- Step 5 (Agent Construction): Reasoning engine
- Step 6 (Tool Layer): Tool registry and execution

**Integration Points**:

#### 1.1 Import LangChain and LLM Models
```python
# src/agent/reasoning.py
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

class ReasoningEngine:
    def __init__(self, llm_provider="openai", model_name="gpt-4"):
        if llm_provider == "openai":
            self.llm = ChatOpenAI(model_name=model_name, temperature=0.3)
        elif llm_provider == "anthropic":
            self.llm = ChatAnthropic(model_name=model_name, temperature=0.3)
        else:
            raise ValueError(f"Unknown provider: {llm_provider}")
```

#### 1.2 Build RAG Memory
```python
# src/agent/memory_rag.py
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MemoryRAG:
    def __init__(self, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        
    def add_documents(self, docs: List[str]):
        """Add documents to vector store"""
        splitter = RecursiveCharacterTextSplitter(chunk_size=500)
        chunks = splitter.split_text(docs)
        self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant documents"""
        return self.vectorstore.similarity_search(query, k=top_k)
```

#### 1.3 Implement Tool Calling
```python
# src/agent/action_generator.py
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StdOutCallbackHandler

class ActionGenerator:
    def __init__(self, tools: List[Tool], llm):
        self.agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            callbacks=[StdOutCallbackHandler()]
        )
    
    def generate(self, state: NetworkState) -> NetworkActions:
        """LLM uses tools to generate actions"""
        prompt = f"""
        Current network state:
        {state.summary()}
        
        Available tools: allocate_power, allocate_spectrum, reconfigure_beam
        
        What actions should we take?
        """
        result = self.agent.run(prompt)
        return parse_actions(result)
```

**File locations to create/modify**:
```
src/agent/
├── reasoning.py           ← LLM reasoning with LangChain
├── memory_rag.py         ← FAISS + embeddings (from RAG-Practice)
└── action_generator.py   ← Tool-calling framework
```

---

### Phase 2: AutoResearch Integration

**What to integrate**: Autonomous experiment loop, hypothesis generation, results tracking

**Where it fits in pipeline**:
- Step 10 (AutoResearch Loop): Autonomous improvement

**Integration Points**:

#### 2.1 Experiment Runner
```python
# research/experiment_runner.py
import subprocess
import git
from pathlib import Path
import time

class AutoResearchRunner:
    def __init__(self, gpu_id=0, training_budget_seconds=300):
        self.gpu_id = gpu_id
        self.training_budget = training_budget_seconds
        self.repo = git.Repo('.')
        self.results_file = Path('research/results.tsv')
        
    def run_experiment(self, hypothesis: str, experiment_id: str):
        """Execute a single experiment with fixed 5-minute budget"""
        
        # Modify code based on hypothesis
        self.apply_hypothesis(hypothesis)
        
        # Commit changes
        self.repo.git.add('train.py')
        self.repo.git.commit(m=f"[autoresearch] Exp {experiment_id}: {hypothesis}")
        
        # Run training with timeout
        start = time.time()
        result = subprocess.run(
            f"CUDA_VISIBLE_DEVICES={self.gpu_id} uv run train.py",
            shell=True,
            capture_output=True,
            timeout=self.training_budget + 60  # Allow 60s overhead
        )
        elapsed = time.time() - start
        
        # Parse results
        val_metric = self.extract_metric(result.stdout, 'val_spectral_eff')
        latency = self.extract_metric(result.stdout, 'latency_ms')
        
        # Log to results.tsv
        self.log_result(
            commit=self.repo.head.commit.hexsha[:7],
            val_metric=val_metric,
            latency=latency,
            status='keep' if val_metric > self.best_metric else 'discard',
            description=hypothesis
        )
        
        return val_metric, latency
    
    def generate_hypothesis(self, failure_analysis: Dict) -> str:
        """Generate new hypothesis from failure analysis"""
        
        # Use LLM to suggest improvements (could be Anthropic Claude)
        from anthropic import Anthropic
        
        client = Anthropic()
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=500,
            system="You are an AI researcher. Based on the failure analysis, suggest ONE specific code change to improve spectral efficiency.",
            messages=[{
                "role": "user",
                "content": f"Recent experiments failed because: {failure_analysis['reason']}\n\nSuggest a specific change to {failure_analysis['file']}:"
            }]
        )
        
        return response.content[0].text
    
    def launch_loop(self, duration_hours=8):
        """Run autonomous research loop"""
        
        start_time = time.time()
        experiment_id = 0
        
        while time.time() - start_time < duration_hours * 3600:
            # Analyze previous failures
            failures = self.analyze_recent_failures()
            
            # Generate hypothesis
            hypothesis = self.generate_hypothesis(failures)
            
            # Run experiment
            val_metric, latency = self.run_experiment(hypothesis, experiment_id)
            
            # Store learnings
            self.update_learnings(
                hypothesis=hypothesis,
                result_metric=val_metric,
                success=val_metric > self.best_metric
            )
            
            experiment_id += 1
            time.sleep(60)  # Wait before next experiment
```

#### 2.2 Results Tracking
```python
# research/experiment_runner.py (continuation)

def log_result(self, commit: str, val_metric: float, latency: float, 
               status: str, description: str):
    """Append result to results.tsv"""
    
    with open(self.results_file, 'a') as f:
        f.write(f"{commit}\t{val_metric:.6f}\t{latency:.1f}\t{status}\t{description}\n")

def update_learnings(self, hypothesis: str, result_metric: float, success: bool):
    """Record insights in learnings.md"""
    
    status = "✅ SUCCESS" if success else "❌ FAILED"
    entry = f"""
## Experiment: {hypothesis}
**Status**: {status}
**Metric**: {result_metric:.4f}
**Timestamp**: {datetime.now().isoformat()}

Key insight: [auto-filled or manually added]
"""
    
    with open('research/learnings.md', 'a') as f:
        f.write(entry + "\n")
```

#### 2.3 Agent Direction Program
```markdown
# research/program.md

The agent should follow this program to direct the AutoResearch loop:

## Goal
Maximize spectral efficiency (current: 5.2 bits/Hz, target: 6.5+)
while maintaining decision latency < 10ms.

## Constraints
- Cannot modify src/optimization/water_filling.py (reference baseline)
- Training must complete in 300 seconds (5 minutes)
- Validation on test scenarios only

## Success History
- Run a1b2c3d: +0.3 bits/Hz by adding adversarial water-filling
- Run d4e5f6g: +0.2 bits/Hz by using DIFFRACT solver
- Run h8i9j0k: -10x latency by NemoIR compilation

## Areas to Explore
1. Channel prediction with transformers (vs LSTM)
2. Jammer detection using attention
3. Hierarchical power allocation (beam-level → frequency-level)
4. Multi-agent RL with user cooperation
5. Uncertainty-aware spectrum allocation

## Next Experiment Ideas
[ ] Add transformer layer for CSI encoding
[ ] Implement beam-search for spectrum bands
[ ] Try multi-task learning (throughput + fairness + latency)
[ ] Add adversarial training with jammer simulator
[ ] Implement curriculum learning (easy → hard scenarios)
```

**File locations to create/modify**:
```
research/
├── program.md              ← Instructions for research agent
├── experiment_runner.py    ← Autonomous loop orchestration
├── results.tsv            ← Experiment results (TSV format)
├── learnings.md           ← Captured insights
└── experiment_logs/       ← Individual experiment outputs
```

---

### Phase 3: NemoIR Integration

**What to integrate**: Workflow compilation, GPU scheduling, graph optimization

**Where it fits in pipeline**:
- Step 7 (Workflow Compilation): NemoIR compiler

**Integration Points**:

#### 3.1 Workflow Definition
```nemo
// src/workflows/agent_workflow.nemo
workflow SpectrumManagementAgent {
    input networkState: NetworkState
    output allocationCommand: AllocationCommand
    
    // Step 1: Perception
    state_vector = perceive_network(networkState)
    
    // Step 2: Parallel predictions (3 GPU workers)
    parallel {
        channel_pred = predict_channel(state_vector)  // GPU 0
        traffic_pred = predict_traffic(state_vector)  // GPU 1
        jammer_diag = detect_jamming(networkState)    // GPU 2
    }
    
    // Step 3: LLM Reasoning (CPU)
    diagnosis = llm_reason(
        channels=channel_pred,
        traffic=traffic_pred,
        jammer=jammer_diag
    )
    
    // Step 4: Tool Selection and Execution
    if diagnosis.strategy == "adversarial_wf" {
        allocation = tool_adversarial_water_filling(
            channels=channel_pred,
            traffic=traffic_pred
        )
    } else if diagnosis.strategy == "diffract" {
        allocation = tool_diffract_optimize(
            channels=channel_pred
        )
    } else {
        allocation = tool_classical_water_filling(
            channels=channel_pred
        )
    }
    
    // Step 5: Reconfiguration (parallel)
    parallel {
        spectrum = tool_allocate_spectrum(allocation)
        beams = tool_reconfigure_beam(networkState)
    }
    
    // Step 6: Command Generation
    command = generate_command(
        power=allocation,
        spectrum=spectrum,
        beams=beams
    )
    
    return command
}
```

#### 3.2 Compilation & Execution
```python
# src/workflows/compile_and_run.py

def compile_workflow(workflow_file: str, output_ir: str):
    """Compile .nemo workflow to IR"""
    
    import subprocess
    result = subprocess.run(
        ['nemoir', 'compile', workflow_file, 
         '--output-ir', output_ir,
         '--optimize=aggressive',
         '--target=gpu'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed: {result.stderr}")
    
    return output_ir

def run_compiled_workflow(ir_file: str, state: NetworkState) -> AllocationCommand:
    """Execute compiled workflow with state input"""
    
    import subprocess
    import json
    
    # Serialize state to JSON
    state_json = json.dumps(state.to_dict())
    
    # Run compiled workflow (binary or Python package)
    result = subprocess.run(
        ['nemoir-runtime', '--ir', ir_file, '--input', state_json],
        capture_output=True,
        text=True,
        timeout=0.01  # 10ms latency budget
    )
    
    # Parse output
    command = AllocationCommand.from_json(result.stdout)
    return command
```

#### 3.3 Performance Monitoring
```python
# src/workflows/profile_workflow.py

def profile_compiled_workflow(ir_file: str, num_iterations: int = 1000):
    """Measure end-to-end latency and throughput"""
    
    import time
    import numpy as np
    
    latencies = []
    
    for i in range(num_iterations):
        state = generate_random_state()
        
        start = time.perf_counter()
        command = run_compiled_workflow(ir_file, state)
        elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
        
        latencies.append(elapsed)
    
    print(f"Compiled Workflow Performance:")
    print(f"  Mean latency: {np.mean(latencies):.2f} ms")
    print(f"  P95 latency:  {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99 latency:  {np.percentile(latencies, 99):.2f} ms")
    print(f"  Throughput:   {1000 / np.mean(latencies):.1f} decisions/sec")
```

**File locations to create/modify**:
```
src/workflows/
├── agent_workflow.nemo      ← Workflow DSL definition
├── compile_and_run.py       ← NemoIR compilation + execution
└── profile_workflow.py      ← Performance monitoring
```

---

## Integration Checklist

### RAG-Practice Integration
- [ ] Copy LangChain/embedding models setup from RAG-Practice
- [ ] Create `src/agent/reasoning.py` with LLM chain
- [ ] Build `src/agent/memory_rag.py` with FAISS vector store
- [ ] Implement tool-calling in `src/agent/action_generator.py`
- [ ] Add domain knowledge documents to `data/knowledge_base/`
- [ ] Test LLM reasoning on sample scenarios

### AutoResearch Integration
- [ ] Create `research/program.md` with research directions
- [ ] Implement `research/experiment_runner.py` with experiment loop
- [ ] Set up results tracking in `research/results.tsv`
- [ ] Create learnings capture in `research/learnings.md`
- [ ] Connect Claude API for hypothesis generation
- [ ] Test 5-minute training budget cycle
- [ ] Set up overnight autonomous experiments

### NemoIR Integration
- [ ] Download NemoIR compiler binary from releases
- [ ] Define `src/workflows/agent_workflow.nemo` DSL
- [ ] Implement compilation in `src/workflows/compile_and_run.py`
- [ ] Verify GPU scheduling and kernel fusion
- [ ] Benchmark compiled vs. sequential latency
- [ ] Create HTML visualization of workflow DAG

## Testing the Integration

### Test RAG-Practice Integration
```bash
# Test LLM reasoning
uv run -c "from src.agent.reasoning import ReasoningEngine; e = ReasoningEngine(); print(e.reason_about_state({'jamming': True}))"

# Test memory retrieval
uv run -c "from src.agent.memory_rag import MemoryRAG; m = MemoryRAG(); m.add_documents(['adversarial water-filling minimizes worst-case jamming']); print(m.retrieve('jamming attack'))"

# Test tool calling
uv run src/agent/action_generator.py --test
```

### Test AutoResearch Integration
```bash
# Initialize experiment tracking
uv run research/experiment_runner.py --init

# Run single experiment
uv run research/experiment_runner.py --hypothesis "add_lstm_layer" --gpu 0

# Monitor results
tail -f research/results.tsv
tail -f research/learnings.md
```

### Test NemoIR Integration
```bash
# Compile workflow
nemoir compile src/workflows/agent_workflow.nemo --output-ir agent.ir --optimize aggressive

# Profile compiled workflow
uv run src/workflows/profile_workflow.py --ir agent.ir

# Visualize DAG
nemoir visualize agent.ir --output agent_graph.html
```

---

## Architecture After Integration

```
AI Spectrum Management (Unified)
│
├── Dataset & Processing
│   └── Sionna wireless simulator + PyTorch/TF
│
├── Optimization & Training
│   ├── CVXPY/DIFFRACT (ground-truth)
│   └── PyTorch/TF models (neural learning)
│
├── Agent Reasoning (RAG-Practice)
│   ├── LangChain + OpenAI/Claude LLM
│   ├── FAISS vector store + embeddings
│   ├── Tool-calling framework
│   └── Structured reasoning with memory
│
├── Autonomous Research (AutoResearch)
│   ├── Experiment runner (5-min budget cycles)
│   ├── Hypothesis generation (Claude)
│   ├── Results tracking (results.tsv)
│   └── Learnings capture (learnings.md)
│
├── Workflow Compilation (NemoIR)
│   ├── NemoIR DSL workflow definition
│   ├── Compiler → Intermediate Representation
│   ├── GPU scheduling + kernel fusion
│   └── Parallel execution (latency: 100ms → 10ms)
│
└── Evaluation & Deployment
    ├── Benchmarking all baselines
    ├── Online real-time execution
    └── Safety monitoring & fallbacks
```

---

## Next Steps

1. **Immediately**: Copy core modules from existing projects
   - `RAG-Practice/src/rag_practice/app.py` → `src/agent/`
   - `autoresearch/train.py` → `research/train_template.py`
   - `nemoir/examples/agent_workflow.nemo` → `src/workflows/`

2. **Short-term**: Implement integration layers
   - Reasoning engine connecting to LLM
   - Experiment runner for autonomous loop
   - Workflow compilation pipeline

3. **Medium-term**: Run integrated experiments
   - Full 10-step pipeline end-to-end
   - Benchmark all baselines
   - Launch overnight AutoResearch

4. **Long-term**: Production deployment
   - Online real-time spectrum allocation
   - Continuous autonomous improvement
   - Deployment to real 6G testbeds

---

## Questions & Support

- **LLM Integration**: See `docs/ARCHITECTURE.md` section on RAG-Practice
- **AutoResearch**: See `research/program.md` and `docs/PIPELINE.md` step 10
- **NemoIR**: See NemoIR documentation at https://github.com/hkalexling/nemoir
- **Wireless**: See Sionna documentation at https://nvlabs.github.io/sionna/
