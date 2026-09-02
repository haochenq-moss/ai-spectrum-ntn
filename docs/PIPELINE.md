# 10-Step AI Agentic Spectrum Management Pipeline

This document provides a **detailed walkthrough of each pipeline stage** with inputs, outputs, algorithms, and integration points.

---

## Step 1: Dataset Building

**Purpose**: Generate realistic NTN wireless environments with channels, traffic, interference, and adversarial jamming.

### 1.1 Scenario & Topology Generation

Create a non-terrestrial network scenario with:
- **Satellite Constellation**: LEO orbit elements, Keplerian dynamics, coverage maps
- **Ground Stations**: User locations, antenna patterns, mobility
- **Ground Infrastructure**: Base stations, relay nodes, network topology
- **Time Evolution**: 24-hour simulations, dynamic topology changes

**Module**: `src/dataset_generation/scenario_topology.py`

**Key Outputs**:
```python
{
    "satellites": [
        {
            "id": "sat_001",
            "position": [x, y, z],  # ECEF coordinates
            "velocity": [vx, vy, vz],
            "antenna": "parabolic_8dbi",
            "orbit": {"inclination": 51.6, "altitude": 550}
        },
        ...
    ],
    "users": [
        {
            "id": "user_001",
            "position": [lat, lon],
            "mobility": "static|vehicular|pedestrian",
            "qos": {"latency_ms": 50, "throughput_mbps": 10}
        },
        ...
    ],
    "link_schedule": [
        {"user": "user_001", "satellite": "sat_001", "duration": 180}
    ]
}
```

### 1.2 Channel Generation

Model wireless propagation channels:
- **Pathloss Models**: Free-space loss, rain attenuation, atmospheric absorption
- **Fading**: Rician/Rayleigh fading for satellite-ground links
- **Doppler Effect**: Frequency shift from satellite motion
- **Multipath**: Reflection from ground, buildings (if applicable)

**Module**: `src/dataset_generation/channel_generation.py`

**Key Functions**:
```python
def generate_channel_state_info(
    scenario: ScenarioConfig,
    frequency_ghz: float,
    time_step: float
) -> ChannelMatrix:
    """
    Returns: 
    - CSI matrix: [n_users × n_satellites × n_subcarriers]
    - Pathloss: [n_users × n_satellites]
    - Phase: Doppler shifts for each link
    """
```

### 1.3 Traffic Generation

Model user traffic patterns:
- **Packet Arrivals**: Poisson, Markov-modulated, realistic traces
- **Packet Sizes**: Fixed, variable, heavy-tailed distributions
- **Burstiness**: ON/OFF models, video streaming, IoT patterns
- **QoS Requirements**: Latency, reliability, throughput per user

**Module**: `src/dataset_generation/traffic_generation.py`

**Output Format**:
```python
{
    "time_step": 1.0,  # seconds
    "users": {
        "user_001": {
            "packets_pending": 10,
            "buffer_size_bits": 50000,
            "qos_class": "URLLC",  # eMBB, URLLC, mMTC
            "priority": 1
        }
    }
}
```

### 1.4 Interference Modeling

Compute multi-user interference:
- **Same-Satellite Interference**: Co-channel users on same beam
- **Adjacent-Satellite Interference**: Adjacent beams, frequency reuse
- **Cross-Link Interference**: Satellite uplink interferes with downlink
- **Ground Interference**: Terrestrial systems, radar, other operators

**Module**: `src/dataset_generation/interference_modeling.py`

**Output**:
```python
interference_matrix  # [n_users × n_subcarriers]
interference_level_dbm  # Power of aggregate interference
```

### 1.5 Adversarial Jammer Generation

Model strategic jamming:
- **Jammer Types**: 
  - Constant-power jammer (static interference)
  - Frequency-selective jammer (narrow-band)
  - Time-variant jammer (pulse, chirp)
  - Intelligent jammer (learns weak channels, adapts)
- **Jammer Placement**: Strategic locations for maximum impact
- **Jamming Power**: Constrained budget, power-control game

**Module**: `src/dataset_generation/adversarial_jammer.py`

**Jammer Definition**:
```python
class AdversarialJammer:
    def __init__(self, power_dbm: float, type: str, channel_state: ChannelState):
        self.power = power_dbm
        self.type = type  # "constant", "sweep", "intelligent"
    
    def allocate_power(self, channels: List[Channel]) -> PowerAllocation:
        """Worst-case power allocation from jammer's perspective"""
        # Returns: jammer_power_per_frequency
```

**Output**:
```python
{
    "jammers": [
        {
            "id": "jammer_001",
            "type": "intelligent",
            "power_dbm": 30,
            "target_bands": [2400, 2410],  # MHz
            "position": [x, y, z]
        }
    ],
    "jamming_interference": jamming_power_matrix  # [n_subcarriers]
}
```

### 1.6 Raw Dataset Assembly

Combine all components into training dataset:
- **Scenario File**: Topology, user locations, satellite orbits
- **Channel Trace**: Time-series CSI for each user-satellite pair
- **Traffic Trace**: Packet arrivals, buffers, QoS
- **Interference Trace**: Thermal noise + multi-user + cross-link
- **Jamming Trace**: Adversarial power allocation vs. time
- **Ground Truth**: Optimal power/spectrum allocation (from Step 3)

**Module**: `src/dataset_generation/dataset_assembler.py`

**Raw Dataset Structure**:
```
data/raw/
├── scenario_001.json          # Topology config
├── channel_trace_001.h5       # CSI time-series (n_timesteps × n_users × n_sats × n_subcarriers)
├── traffic_001.csv            # User packets, buffers
├── interference_001.npy       # Interference + noise
├── jamming_001.npy            # Adversarial jamming profile
└── labels_001.npy             # Optimal action (from Step 3)
```

**Size**: ~1-10 GB per scenario, 100+ scenarios = 100+ GB dataset

---

## Step 2: Data Processing & State Representation

**Purpose**: Convert raw simulation data into standardized learning states for neural networks.

### 2.1 Data Cleaning & Validation

- **Outlier Detection**: Flag unrealistic CSI values, infinite losses
- **Missing Data Handling**: Interpolate or forward-fill time gaps
- **Consistency Checks**: Symmetry of channel matrices, energy conservation
- **Normalization**: Map values to [0, 1] or z-score

**Module**: `src/data_processing/cleaning.py`

### 2.2 Feature Engineering

Extract meaningful features from raw data:

**Channel Features**:
- SINR per user-satellite-subcarrier: $\frac{P_{\text{signal}}}{P_{\text{noise+interference}}}$
- Channel gain (dB), pathloss, Doppler shift
- Channel coherence time, delay spread
- CSI quantization levels (4-bit, 8-bit)

**Traffic Features**:
- Buffer occupancy (bits, packets)
- Arrival rate (packets/sec)
- QoS urgency (time-to-deadline)
- Traffic class (eMBB, URLLC, mMTC)

**Interference Features**:
- Aggregate interference power (dBm)
- Interference-to-noise ratio (INR)
- Number of interferers
- Interference burstiness

**Jammer Features**:
- Jammer power (dBm)
- Jammer type (encoded)
- Jammer presence indicator (binary)
- Predicted jammer strategy (RNN-based)

**Module**: `src/data_processing/feature_engineering.py`

**Output**:
```python
feature_vector = {
    "sinr_db": np.array([15, 22, 18, ...]),  # SINR per channel
    "channel_gain_linear": np.array([...]),
    "buffer_occupancy_bits": 50000,
    "traffic_priority": 1,
    "interference_power_dbm": 20,
    "jammer_detected": True,
    "jammer_power_dbm": 30,
    "timestamp": 1234567890
}
```

### 2.3 Channel/Topology Encoding

Encode network topology and channel state as compact vectors:

**Topology Encoding** (learned embeddings):
- User position embedding: sin/cos positional encoding
- Satellite visibility: binary adjacency matrix
- Beam pattern: neural encoding of antenna pattern
- Network graph: Graph Neural Network (GNN) embedding

**Channel Encoding**:
- CSI vector: Compress 1000+ subcarriers → 64-dim representation
- Quantization: Mimic real CSI feedback (4-8 bit)
- Temporal encoding: RNN to capture channel evolution

**Module**: `src/data_processing/feature_engineering.py`

### 2.4 Traffic & Jammer Feature Extraction

Extract time-series features:

**Traffic Time-Series** (last 10 packets):
- Packet inter-arrival times
- Packet sizes
- Queue depth
- Deadline urgency

**Jammer Time-Series** (last 10 observations):
- Jamming power vs. time
- Frequency hopping pattern
- Adaptation to network (if intelligent)
- Predicted next action

**Module**: `src/data_processing/feature_engineering.py`

### 2.5 State Construction

Combine all features into standardized **state vector** $\mathbf{s}$:

$$\mathbf{s} = [\text{topology\_embedding}, \text{channel\_encoding}, \text{traffic\_features}, \text{interference\_features}, \text{jammer\_features}]$$

**Typical Dimensions**: 256-512 float32 values

```python
class NetworkState:
    def __init__(self):
        self.topology_embedding: np.ndarray  # [64]
        self.channel_state: np.ndarray       # [128] - CSI compression
        self.traffic_features: np.ndarray    # [32]
        self.interference_power: np.ndarray  # [8]
        self.jammer_profile: np.ndarray      # [16]
        self.timestamp: float
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to PyTorch tensor for neural networks"""
        return torch.cat([
            self.topology_embedding,
            self.channel_state,
            ...
        ]).unsqueeze(0)  # [1, 256]
```

**Module**: `src/data_processing/state_construction.py`

### 2.6 Train/Validation/Test Split

Partition dataset respecting **temporal coherence**:
- **Train** (70%): Earlier scenarios, used for supervised learning
- **Validation** (15%): Intermediate scenarios, hyperparameter tuning
- **Test** (15%): Future scenarios (not seen during training)

**Stratification**:
- Balanced across jamming levels (no jammer, weak, strong, intelligent)
- Balanced across satellite density
- No temporal leakage (train timesteps < test timesteps)

**Module**: `src/data_processing/data_splitting.py`

---

## Step 3: Ground-Truth Optimization

**Purpose**: Generate optimal/near-optimal solutions to supervise learning models.

### 3.1 Problem Formulation

**Standard Spectrum Allocation Problem**:

Maximize: $\sum_u \log_2(1 + \text{SINR}_u)$ (Shannon capacity)

Subject to:
- Power constraints: $\sum_u P_{u,f} \leq P_{\text{max}}$ per subcarrier
- Interference: $\text{SINR}_u = \frac{P_{u,f} |h_{u,f}|^2}{I_{u,f} + N_0}$
- Rate constraints: $R_u \geq R_{\min,u}$ (QoS)
- Spectrum allocation: Binary frequency assignment

**Adversarial Water-Filling (Minimax)**:

$$\max_{\mathbf{P}} \min_{\text{jammer}} \sum_u \log_2(1 + \text{SINR}_u(\mathbf{P}, \text{jammer}))$$

This ensures **robustness** against worst-case jamming.

**Module**: `src/optimization/adversarial_water_filling.py`

### 3.2 Adversarial Water-Filling

Solve the minimax game iteratively:

1. **Outer loop (resource allocator)**: Allocate power $\mathbf{P}$ to maximize capacity
2. **Inner loop (jammer)**: Find worst-case jamming that minimizes capacity
3. **Iterate**: Use game theory (support enumeration, linear programming) to find equilibrium

**Algorithm**:
```python
def adversarial_water_filling(
    channels: np.ndarray,          # [n_users × n_subcarriers]
    noise_power: float,
    power_budget: float,
    jammer_budget: float,
    max_iterations: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
    - power_allocation: [n_users × n_subcarriers]
    - jammer_allocation: [n_subcarriers]
    """
    power = np.ones_like(channels) * power_budget / channels.size
    jammer = np.ones(channels.shape[1]) * jammer_budget / channels.shape[1]
    
    for iteration in range(max_iterations):
        # Allocator's turn: water-filling against current jammer
        sinr = compute_sinr(channels, power, jammer, noise_power)
        power = water_filling_step(channels, sinr, power_budget)
        
        # Jammer's turn: attack worst user
        sinr = compute_sinr(channels, power, jammer, noise_power)
        jammer = jammer_attack_step(channels, power, jammer_budget)
        
        if converged(power, jammer):
            break
    
    return power, jammer
```

**Module**: `src/optimization/adversarial_water_filling.py`

### 3.3 DIFFRACT Optimization

Use **D**ifferentiable optimization for resource allocation:

DIFFRACT (Differentiable Framework for Resource Allocation and Channel Training) solves:

$$\min_{\mathbf{P}} \left[ \sum_u (-\log_2(1 + \text{SINR}_u(\mathbf{P}))) + \lambda \|\mathbf{P}\|_1 \right]$$

Subject to: Convex constraints on power, interference, spectrum

**Advantages**:
- Differentiable → gradient-based optimization (fast)
- End-to-end training possible
- Generalizes to unseen problem sizes

**Implementation**:
```python
import cvxpy as cp

def diffract_optimize(channels, noise_power, power_budget):
    n_users, n_subcarriers = channels.shape
    P = cp.Variable((n_users, n_subcarriers), nonneg=True)  # Power matrix
    
    # Objective: maximize total capacity
    sinr = cp.multiply(P, channels) / (noise_power + 0.1)  # Approximate
    capacity = cp.sum(cp.log2(1 + sinr))
    
    # Constraints
    constraints = [
        cp.sum(P, axis=0) <= power_budget,  # Power per subcarrier
        P >= 0,
        P <= power_budget / n_users  # Upper bound per user
    ]
    
    problem = cp.Problem(cp.Maximize(capacity), constraints)
    problem.solve(solver=cp.SCS)
    
    return P.value
```

**Module**: `src/optimization/diffract_optimizer.py`

### 3.4 Constraint Handling

Enforce network constraints:

- **Power Constraints**: $\sum_u P_u \leq P_{\max}$ (satellite TX power limit)
- **Interference Constraints**: $\text{SINR}_u \geq \text{SINR}_{\min}$ (QoS threshold)
- **Spectrum Constraints**: Each user assigned contiguous or discrete bands
- **Fair Allocation**: Proportional fair or max-min fair rates
- **Latency Constraints**: Decision must complete in < 10ms

**Module**: `src/optimization/constraint_handling.py`

### 3.5 Teacher Dataset Generation

For each scenario in training set, compute and store optimal action:

```python
def generate_teacher_labels(dataset_config):
    """Generate (state, optimal_action) pairs"""
    
    optimal_actions = []
    
    for scenario_id in range(num_scenarios):
        # Load scenario state
        state = load_state(scenario_id)
        channels = state.channels
        traffic = state.traffic
        interference = state.interference
        
        # Compute optimal action using ground-truth optimization
        if use_adversarial_wf:
            power_alloc, _ = adversarial_water_filling(channels, ...)
        elif use_diffract:
            power_alloc = diffract_optimize(channels, ...)
        else:
            power_alloc = classical_water_filling(channels, ...)
        
        # Store (state, action) pair
        optimal_actions.append({
            "state": state.to_vector(),
            "action": power_alloc,
            "reward": compute_throughput(power_alloc, state)
        })
    
    return optimal_actions
```

**Output Dataset**:
```
data/ground_truth/
├── teacher_states.h5       # [n_scenarios × 256] state vectors
├── teacher_actions.h5      # [n_scenarios × n_users × n_subcarriers]
└── teacher_rewards.h5      # [n_scenarios] capacity achieved
```

**Module**: `src/optimization/teacher_dataset.py`

---

## Step 4: Neural Model Training

**Purpose**: Learn fast approximations of expensive optimization algorithms.

### 4.1 Supervised Learning (Imitation)

**Goal**: Train neural network to imitate teacher solutions

$$\mathcal{L}_{\text{supervised}} = \mathbb{E}_{(s, a^*)} \left[ \|\hat{a}(s) - a^*\|_2^2 \right]$$

Where $a^*$ comes from Step 3 (teacher solutions).

**Architecture**: Deep neural network
- Input: State vector (256-dim)
- Hidden: 3-4 dense layers (256 → 128 → 64 → 32)
- Output: Power allocation (n_users × n_subcarriers)
- Activation: ReLU → Sigmoid (output bounded [0,1])

```python
class SupervisedPolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()  # Output in [0, 1]
        )
    
    def forward(self, state):
        return self.net(state)
```

**Training**:
```python
def train_supervised_model(teacher_states, teacher_actions, epochs=100):
    model = SupervisedPolicyNetwork(state_dim=256, action_dim=n_actions)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        for batch_states, batch_actions in dataloader:
            predicted = model(batch_states)
            loss = F.mse_loss(predicted, batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if epoch % 10 == 0:
            val_loss = evaluate_on_validation(model)
            print(f"Epoch {epoch}: Val MSE = {val_loss:.4f}")
```

**Module**: `src/models/supervised_learning.py`

### 4.2 Reinforcement Learning

**Goal**: Learn online policy through trial-and-error with environment rewards

**Algorithm**: Proximal Policy Optimization (PPO)

$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t \left[ \min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t) \right]$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$ is importance ratio.

**Agent Architecture**:
- **Actor**: Maps state → power allocation (continuous control)
- **Critic**: Estimates value function $V(s)$
- **Environment**: Wireless simulator (Sionna-based)
- **Reward**: Throughput (capacity) or fairness metric

```python
class PPOAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        action_mean = self.actor(state)
        action_std = self.log_std.exp()
        dist = torch.distributions.Normal(action_mean, action_std)
        return dist, self.critic(state)
```

**Training Loop**:
```python
def train_ppo(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        
        for t in range(max_steps):
            with torch.no_grad():
                dist, value = agent(state)
                action = dist.sample()
            
            next_state, reward, done = env.step(action)
            store_transition(state, action, reward, value)
            
            state = next_state
            episode_reward += reward
            
            if done:
                break
        
        # PPO update
        update_ppo(agent, stored_transitions)
        
        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.2f}")
```

**Module**: `src/models/reinforcement_learning.py`

### 4.3 Predictive Models

Train separate models for channel, traffic, and jammer prediction:

**Channel Predictor** (LSTM/Transformer):
- Input: Past 10 CSI observations
- Output: Next-step CSI
- Loss: MSE on magnitude + phase

**Traffic Predictor**:
- Input: Packet history (sizes, inter-arrivals)
- Output: Next packet arrival, size
- Loss: Cross-entropy for arrival class, MSE for size

**Jammer Detector**:
- Input: Received signal spectrum
- Output: Jammer present (binary) + jammer type (multiclass) + power (regression)
- Loss: Binary crossentropy + cross-entropy + MSE

**Module**: `src/models/channel_predictor.py`, `traffic_predictor.py`, `jammer_detector.py`

### 4.4 Uncertainty Estimation

Add uncertainty quantification for risk-aware decisions:

**Bayesian Neural Networks** (MC Dropout):
```python
class BayesianPolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, dropout_rate=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
    
    def forward(self, state, num_samples=100):
        # MC Dropout: sample from posterior
        predictions = []
        for _ in range(num_samples):
            predictions.append(self.net(state))
        
        mean = torch.mean(torch.stack(predictions), dim=0)
        std = torch.std(torch.stack(predictions), dim=0)
        return mean, std
```

**Module**: `src/models/uncertainty_estimator.py`

---

## Step 5: AI Agent Construction

**Purpose**: Build autonomous agent that reasons about network conditions and orchestrates tools.

### 5.1 Environment Perception

Observe current network state:

```python
class EnvironmentPerception:
    def observe(self) -> EnvironmentSnapshot:
        """Real-time observation of network"""
        return {
            "channel_state": self.measure_csi(),
            "user_traffic": self.query_buffer_status(),
            "interference": self.estimate_interference(),
            "jammer_activity": self.detect_jamming(),
            "timestamp": time.time()
        }
```

**Measurement Methods**:
- **CSI**: Pilot-based estimation (sounding)
- **Traffic**: Queue depth, arrival rate
- **Interference**: Energy detection, spectral analysis
- **Jamming**: Signal classification, anomaly detection

**Module**: `src/agent/environment_perception.py`

### 5.2 State Understanding

Convert observations into standardized state vector:

```python
class StateUnderstanding:
    def interpret(self, observation: EnvironmentSnapshot) -> NetworkState:
        """Convert observation to state vector"""
        
        # Encode topology, channels, traffic, interference (from Step 2)
        state_vector = self.encoder(
            observation.channel_state,
            observation.user_traffic,
            observation.interference,
            observation.jammer_activity
        )
        
        return NetworkState(vector=state_vector, observation=observation)
```

**Module**: `src/agent/state_understanding.py`

### 5.3 LLM-Based Reasoning (RAG-Practice)

Use Large Language Model to reason about network conditions:

```python
class ReasoningEngine:
    def __init__(self, llm_client, memory_rag):
        self.llm = llm_client  # OpenAI, Claude, etc.
        self.memory = memory_rag  # FAISS + embeddings
    
    def reason(self, state: NetworkState) -> Diagnosis:
        """LLM diagnoses network conditions"""
        
        # Retrieve relevant domain knowledge
        context = self.memory.retrieve(
            query=f"Network condition: {state.summary()}",
            top_k=5
        )
        
        # Construct prompt for LLM
        prompt = f"""
        Current Network State:
        - SINR per user: {state.sinr_db}
        - Buffer occupancy: {state.buffer_occupancy}
        - Detected jammer: {state.jammer_detected}
        
        Relevant Knowledge:
        {context}
        
        Question: What is the likely cause of throughput degradation? 
        Which optimization strategy should we use (water-filling, DIFFRACT, RL)?
        """
        
        response = self.llm.complete(prompt)
        diagnosis = parse_diagnosis(response)
        
        return diagnosis
```

**Diagnosis Output**:
```python
{
    "network_condition": "adversarial_jamming_detected",
    "severity": "high",
    "recommended_strategy": "adversarial_water_filling",
    "confidence": 0.92,
    "reasoning": "Strong narrowband interference detected on primary channel..."
}
```

**Module**: `src/agent/reasoning.py`
**Integration**: RAG-Practice (LangChain + OpenAI/Claude API)

### 5.4 Planning & Tool Selection

Agent decides which tools to call:

```python
class Planner:
    def plan(self, diagnosis: Diagnosis, available_tools: ToolRegistry) -> Plan:
        """Decompose diagnosis into tool calls"""
        
        plan = []
        
        # Tool 1: Estimate channel (if needed)
        if diagnosis.requires_channel_update:
            plan.append(ToolCall(
                tool="estimate_channel",
                args={"pilots": 100, "method": "linear"}
            ))
        
        # Tool 2: Detect jamming (if needed)
        if diagnosis.jammer_activity:
            plan.append(ToolCall(
                tool="detect_jamming",
                args={"sensitivity": "high"}
            ))
        
        # Tool 3: Allocate power (main action)
        if diagnosis.recommended_strategy == "adversarial_water_filling":
            plan.append(ToolCall(
                tool="allocate_power",
                args={"method": "adversarial_wf", "iterations": 20}
            ))
        elif diagnosis.recommended_strategy == "diffract":
            plan.append(ToolCall(
                tool="allocate_power",
                args={"method": "diffract", "convergence": 1e-3}
            ))
        else:  # fallback to simple WF
            plan.append(ToolCall(
                tool="allocate_power",
                args={"method": "classical_wf"}
            ))
        
        # Tool 4: Allocate spectrum
        plan.append(ToolCall(
            tool="allocate_spectrum",
            args={"fairness": "proportional"}
        ))
        
        # Tool 5: Reconfigure beams
        plan.append(ToolCall(
            tool="reconfigure_beam",
            args={"beam_type": "adaptive", "resolution": "fine"}
        ))
        
        return plan
```

**Module**: `src/agent/ai_agent.py` (Planner class)

### 5.5 Memory & RAG System

Build knowledge base for agent reasoning:

```python
class MemoryRAG:
    def __init__(self, embedding_model, faiss_index):
        self.embeddings = embedding_model  # sentence-transformers
        self.index = faiss_index  # FAISS vector DB
        self.documents = []  # Original texts
    
    def add_knowledge(self, doc: str, category: str):
        """Add domain knowledge to memory"""
        
        # Split into chunks
        chunks = split_into_chunks(doc, chunk_size=500)
        
        for chunk in chunks:
            # Embed chunk
            embedding = self.embeddings.encode(chunk)
            
            # Store in index
            self.index.add(np.array([embedding]))
            self.documents.append({
                "text": chunk,
                "category": category,
                "timestamp": time.time()
            })
    
    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant knowledge"""
        
        # Embed query
        query_embedding = self.embeddings.encode(query)
        
        # Search index
        distances, indices = self.index.search(
            np.array([query_embedding]), 
            k=top_k
        )
        
        # Return top matches
        results = [self.documents[i]["text"] for i in indices[0]]
        return results
```

**Knowledge Base Contents**:
- 6G spectrum regulations and standards
- Adversarial game theory (water-filling, Nash equilibrium)
- Jamming detection techniques
- Satellite communication principles
- Previous successful allocations (case studies)

**Module**: `src/agent/memory_rag.py`
**Integration**: RAG-Practice (LangChain + FAISS + embeddings)

### 5.6 Action Generation

Convert tool calls and tool outputs into concrete network commands:

```python
class ActionGenerator:
    def generate_actions(self, tool_results: List[ToolResult]) -> NetworkActions:
        """Convert tool outputs to network reconfiguration commands"""
        
        power_allocation = tool_results["allocate_power"]
        spectrum_allocation = tool_results["allocate_spectrum"]
        beam_config = tool_results["reconfigure_beam"]
        
        # Construct network commands
        actions = {
            "power_command": PowerCommand(allocation=power_allocation),
            "spectrum_command": SpectrumCommand(bands=spectrum_allocation),
            "beam_command": BeamCommand(weights=beam_config),
            "handover_command": HandoverCommand(user_sat_pairs=self.compute_handovers(spectrum_allocation))
        }
        
        return actions
```

**Module**: `src/agent/action_generator.py`

---

## Step 6: Agent Tool Layer

**Purpose**: Provide reliable wireless functions for agent tool calls.

See [API_REFERENCE.md](API_REFERENCE.md) for complete tool definitions.

**Key Tools**:

| Tool | Input | Output | Example |
|------|-------|--------|---------|
| `estimate_channel()` | Pilot signals | CSI matrix | CSI for 100 subcarriers |
| `detect_jamming()` | Received signal | Jammer profile | Type: narrowband, Power: 30dBm |
| `predict_traffic()` | History | Next demand | 50 packets expected in 100ms |
| `allocate_power()` | Budget, channels | Power matrix | 10W per user optimized |
| `allocate_spectrum()` | Demand, interference | Spectrum bands | User 1: 2400-2410 MHz |
| `reconfigure_beam()` | Target direction | Beamweights | Complex weights for 64 antenna |
| `handover_user()` | User, options | New satellite | Switch to satellite_002 |
| `evaluate_policy()` | State, action | Reward | Throughput: 50 Mbps, Fairness: 0.8 |

**Module**: `src/tools/`

---

## Step 7: Agent Workflow & NemoIR Compilation

**Purpose**: Compile agent from sequential FSM into optimized DAG for low-latency execution.

### 7.1 Workflow Definition (`.nemo`)

Define agent as NemoIR workflow:

```nemo
// Agent workflow in NemoIR DSL
workflow SpectrumManagementAgent {
    input networkState: NetworkState
    output allocationCommand: AllocationCommand
    
    // Perception
    state_vector = interpret(networkState)
    
    // Reasoning (parallel: channel + traffic + jammer prediction)
    parallel {
        channel_prediction = predict_channel(state_vector)
        traffic_forecast = predict_traffic(state_vector)
        jammer_diagnosis = detect_jamming(networkState.signal)
    }
    
    // LLM reasoning about diagnosis
    diagnosis = llm_reason(
        channel_pred=channel_prediction,
        traffic_pred=traffic_forecast,
        jammer=jammer_diagnosis
    )
    
    // Tool selection based on diagnosis
    if diagnosis.strategy == "adversarial_wf" {
        allocation = adversarial_water_filling(
            channels=channel_prediction,
            traffic=traffic_forecast
        )
    } else if diagnosis.strategy == "diffract" {
        allocation = diffract_optimize(
            channels=channel_prediction,
            traffic=traffic_forecast
        )
    } else {
        allocation = classical_water_filling(
            channels=channel_prediction
        )
    }
    
    // Spectrum and beam reconfiguration (parallel)
    parallel {
        spectrum = allocate_spectrum(allocation)
        beams = reconfigure_beam(
            targets=extract_user_targets(networkState),
            channels=channel_prediction
        )
    }
    
    // Generate network command
    command = generate_network_command(
        power=allocation,
        spectrum=spectrum,
        beams=beams
    )
    
    return command
}
```

**Module**: `src/workflows/agent_workflow.nemo`

### 7.2 NemoIR Compilation

Compile workflow to Intermediate Representation:

```bash
nemoir compile src/workflows/agent_workflow.nemo \
    --output-ir agent_workflow.ir \
    --optimize aggressive \
    --target gpu
```

**Compilation Output** (`agent_workflow.ir`):
- **DAG**: Directed acyclic graph of operations
- **Parallelism**: Marks for independent execution
- **GPU Scheduling**: Schedules numeric operations (water-filling, DIFFRACT) on GPU
- **Latency Estimate**: Predicts decision latency

### 7.3 Graph Optimization

NemoIR optimizer performs:
- **Common Subexpression Elimination**: Avoid redundant computations
- **Kernel Fusion**: Merge adjacent GPU operations
- **Memory Layout**: Optimize tensor layouts for cache efficiency
- **Dependency Analysis**: Maximize parallelism

### 7.4 Parallel Execution

Compiled workflow enables:
- **Channel/Traffic/Jammer predictions** run in parallel (GPU)
- **Water-Filling and DIFFRACT** run on GPU cores
- **Beam reconfiguration** parallelized across antenna elements

**Speed Gain**: 100ms (sequential) → 10ms (compiled + parallel)

### 7.5 GPU Scheduling

NemoIR schedules numeric operations on available GPUs:

```
Timeline (compiled execution):
0ms:     receive_state
         ├─ predict_channel (GPU 0)
         ├─ predict_traffic (GPU 1)
         └─ detect_jamming  (GPU 0)
2ms:     all predictions ready
         ├─ llm_reason      (GPU 2, CPU-based)
3ms:     reasoning complete
         ├─ adversarial_wf  (GPU 0) [5ms]
         └─ allocate_spectrum (GPU 1) [2ms]
8ms:     optimization complete
         ├─ reconfigure_beam (GPU 0) [1ms]
9ms:     all done, return command
```

**Module**: `src/workflows/`
**Integration**: NemoIR compiler + backend

---

## Step 8: Evaluation & Benchmarking

**Purpose**: Measure and compare approaches on standardized metrics.

### Metrics

**Spectral Efficiency** (bits/Hz/s):
$$\eta = \frac{\sum_u R_u}{B}$$
where $B$ is bandwidth.

**Latency**:
- **Decision Latency**: Time from observation to action (< 10ms target)
- **End-to-End Latency**: Observation → reconfiguration → effect (< 50ms)

**Fairness** (Jain's Index):
$$J = \frac{(\sum_u R_u)^2}{n \sum_u R_u^2}$$
(0=unfair, 1=perfectly fair)

**Energy Efficiency** (bits/Joule):
$$\nu = \frac{\sum_u R_u}{\sum_u P_u}$$

**Robustness to Jamming**:
- Throughput degradation under jamming vs. no-jammer baseline
- Recovery time after jamming stops

**Outage Probability**:
$$P_{\text{out}} = P(\text{SINR} < \text{SINR}_{\min})$$

### Baseline Comparison

| Baseline | Method | Optimality | Latency | Complexity |
|----------|--------|-----------|---------|-----------|
| 1. Water-Filling | Classical algorithm | Optimal (no jammer) | 50-100ms | Low |
| 2. AWF | Minimax game theory | Robust to worst-case | 100-200ms | Medium |
| 3. DIFFRACT | Gradient-based optimization | Near-optimal | 10-50ms | Medium |
| 4. RL Agent | PPO policy | Learned from experience | 1-5ms | High |
| 5. LLM Agent | Reasoning + tool calls | Interpretable | 20-50ms | High |
| 6. **Proposed**: Agent + DIFFRACT | LLM reasoning + numerical solver | Robust + fast | 15-25ms | High |
| 7. **Proposed**: NemoIR-Compiled | Compiled workflow | Optimized execution | **5-10ms** | Very High |

### Benchmarking Code

```python
def benchmark(scenarios, baselines, metrics):
    """Run comprehensive benchmark"""
    
    results = {}
    
    for baseline_name, baseline_fn in baselines.items():
        baseline_results = {metric: [] for metric in metrics}
        
        for scenario in scenarios:
            state = scenario.initial_state
            
            # Run baseline
            start_time = time.time()
            action = baseline_fn(state)
            latency = (time.time() - start_time) * 1000  # ms
            
            # Simulate outcome
            next_state, reward = scenario.step(action)
            
            # Compute metrics
            spectral_eff = compute_spectral_efficiency(next_state)
            fairness = compute_fairness(next_state)
            energy_eff = compute_energy_efficiency(next_state)
            
            baseline_results["spectral_efficiency"].append(spectral_eff)
            baseline_results["latency_ms"].append(latency)
            baseline_results["fairness"].append(fairness)
            baseline_results["energy_efficiency"].append(energy_eff)
        
        # Aggregate results
        results[baseline_name] = {
            metric: {
                "mean": np.mean(baseline_results[metric]),
                "std": np.std(baseline_results[metric]),
                "min": np.min(baseline_results[metric]),
                "max": np.max(baseline_results[metric])
            }
            for metric in metrics
        }
    
    return results
```

**Module**: `src/evaluation/benchmarks.py`

---

## Step 9: Online NTN Execution

**Purpose**: Deploy agent for real-time spectrum allocation in live networks.

### Execution Loop

```python
class OnlineExecutor:
    def __init__(self, agent, compiled_workflow):
        self.agent = agent
        self.workflow = compiled_workflow
        self.state_buffer = []
    
    def run_forever(self):
        """Real-time spectrum allocation loop"""
        
        while True:
            # Step 1: Observe network
            observation = observe_network_state()
            self.state_buffer.append(observation)
            
            # Step 2: Estimate state
            state = self.filter_state(self.state_buffer[-10:])
            
            # Step 3: Run compiled workflow
            start_time = time.time()
            action = self.workflow.execute(state)
            decision_latency = time.time() - start_time
            
            # Step 4: Apply action
            try:
                apply_network_config(action)
            except Exception as e:
                # Fallback to conservative allocation
                apply_fallback_config()
                log_error(e)
            
            # Step 5: Evaluate outcome
            next_observation = observe_network_state()
            reward = compute_reward(next_observation)
            
            # Step 6: Log results
            log_decision(state, action, reward, decision_latency)
            
            # Sleep until next control interval
            time.sleep(max(0, control_interval - decision_latency))
```

**Control Interval**: Typically 100-500ms (depends on channel coherence time)

### Safety & Fallbacks

- **Confidence Threshold**: Only apply AI decision if confidence > 0.8
- **Conservative Fallback**: Default to equal-power allocation if uncertain
- **Monitoring**: Continuously check if applied allocation makes conditions worse
- **Human-in-Loop**: Alert operator if unusual pattern detected

**Module**: `src/evaluation/online_executor.py`

---

## Step 10: Autoresearch Loop

**Purpose**: Continuously improve algorithms without human intervention.

### Loop Mechanism

```
REPEAT FOREVER:
  1. Observe Results
     → Which scenarios failed?
     → What was the error magnitude?
  
  2. Identify Failure Cases
     → Jamming-heavy scenarios?
     → Low-SNR conditions?
     → High-traffic bursts?
  
  3. Generate New Hypothesis
     → Change model architecture? (add LSTM layer)
     → Tune hyperparameters? (learning rate, batch size)
     → Expand dataset? (add more jamming scenarios)
     → Try new algorithm? (switch to AWF)
  
  4. Modify Code
     → Agent edits train.py or models/
     → Adds new feature engineering
     → Changes loss function
  
  5. Train
     → GPU training with 5-minute budget
     → Compute val metric (spectral efficiency, latency)
  
  6. Evaluate
     → Compare to previous best
     → Check fairness, robustness
  
  7. Log Result
     → Commit to git with description
     → Record in results.tsv (keep/discard)
     → Update learnings.md
  
  8. Repeat
```

### Research Program

Agent follows instructions in `research/program.md`:

```markdown
# AutoResearch Program for Spectrum Management

## Goal
Improve spectral efficiency (bits/Hz/s) from 5.2 to 6.5+
while keeping decision latency < 10ms.

## Constraints
- Cannot modify prepare.py (fixed)
- Cannot install new packages
- Training must complete in < 5 minutes per run
- Must validate on test set before committing

## Successful Areas (from learnings.md)
- Adversarial Water-Filling: +0.3 bits/Hz
- DIFFRACT optimization: +0.2 bits/Hz
- LLM reasoning: interpretability bonus
- NemoIR compilation: -10x latency

## Areas to Explore
- Channel prediction with transformers
- Jammer detection using attention mechanisms
- Hierarchical resource allocation (beam-level then frequency)
- Multi-agent learning (cooperative users)

## Experiment Ideas (pick one per run)
1. Add transformer layer for channel prediction
2. Use adversarial jamming in training data
3. Switch from PPO to TRPO for RL agent
4. Add fairness constraint to optimization
5. Implement beam-search for spectrum allocation
```

### Experiment Tracking

Each experiment recorded in `results.tsv`:

```
commit          val_spectral_eff   latency_ms   status    description
a1b2c3d         5.200             100.0        keep      baseline
b2c3d4e         5.350             95.2         keep      + adversarial WF
c3d4e5f         5.380             88.5         keep      + DIFFRACT solver
d4e5f6g         5.200             101.0        discard   transformer too slow
e5f6g7h         5.420             82.0         keep      + channel predictor LSTM
f6g7h8i         5.250             0.0          crash     OOM in jammer detector
g7h8i9j         5.510             75.0         keep      + LLM reasoning
h8i9j0k         5.650             68.0         keep      + NemoIR compilation
```

### Learning Capture

Insights recorded in `research/learnings.md`:

```markdown
# AutoResearch Learnings Log

## Key Finding 1: Adversarial WF Critical for Robustness
- Experiments d4e5f6g–h8i9j0k show AWF consistently outperforms classical WF
- Improvement holds across all jammer types
- Recommendation: Make AWF the default strategy

## Key Finding 2: NemoIR Compilation Enables Real-Time
- Baseline latency: 100ms (sequential reasoning + optimization)
- After compilation: 10ms (parallel execution + GPU scheduling)
- Enables sub-50ms online decision-making
- Recommendation: Always compile agent workflow before deployment

## Hypothesis 3: Channel Prediction via Transformer
- Attempt: e5f6g7h (failed)
- Attempt: g7h8i9j (success with LSTM, +0.15 bits/Hz)
- Try: Attention-based transformer next run
- Note: Need to handle variable sequence lengths
```

**Module**: `research/`
**Integration**: AutoResearch framework + Anthropic/Claude API

---

## Integration Summary

```
Dataset Generation (Step 1)
    ↓ Raw data (channels, traffic, interference, jammer)
    ├→ Data Processing (Step 2)
    │   ↓ Cleaned, featured states
    │   └→ Optimization (Step 3)
    │       ↓ Optimal actions (teacher labels)
    │       └→ Model Training (Step 4)
    │           ↓ Trained neural networks
    │           └→ Agent Construction (Step 5)
    │               ↓ Reasoning engine + planning
    │               └→ Tools Layer (Step 6)
    │                   ↓ Wireless functions
    │                   └→ Workflow Compilation (Step 7)
    │                       ↓ NemoIR IR + GPU scheduling
    │                       ├→ Evaluation (Step 8)
    │                       │   ↓ Performance metrics
    │                       │   ↓ Baseline comparison
    │                       └→ Online Execution (Step 9)
    │                           ↓ Live spectrum allocation
    │                           ↓ Safety monitoring
    │
    └→ AutoResearch Loop (Step 10) ──┐
        ↑ Observe failures            │
        ↑ Generate hypotheses         │
        ↑ Modify code (train.py, etc) │
        ↑ Train 5-minute budget       │
        ↑ Keep/discard improvement    │
        └─────────────────────────────┘
```

All steps use the **three integrated frameworks**:
- **RAG-Practice**: Steps 2, 5-6 (memory, reasoning, tools)
- **AutoResearch**: Step 10 (autonomous loop)
- **NemoIR**: Step 7 (workflow compilation for real-time)

---

**Next**: See [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) for how to run experiments.
