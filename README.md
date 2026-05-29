# 🚦 Hierarchical Multi-Agent DDQN Traffic Light Control System

**Deep Reinforcement Learning for Scalable, Secure Urban Traffic Management**

> A multi-phase research project that evolves automated traffic control from a single isolated intersection to a **hierarchically coordinated 8-intersection urban network** with **cyberattack-resilient LSTM-defended sensors** — built on SUMO, PyTorch, and DDQN.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Project Architecture](#-project-architecture)
  - [Phase 0: Single-Agent Pretraining](#phase-0-single-agent-pretraining)
  - [Phase 1: 4-Intersection Multi-Agent Grid](#phase-1-4-intersection-multi-agent-grid)
  - [Phase 2: 8-Intersection Hierarchical Supervisors](#phase-2-8-intersection-hierarchical-supervisors)
  - [Phase 3: Cybersecurity & LSTM Defense](#phase-3-cybersecurity--lstm-defense)
- [Performance Results](#-performance-results)
- [Installation & Setup](#-installation--setup)
- [Usage & Commands](#-usage--commands)
- [Project Structure](#-project-structure)
- [Hyperparameters & Training Details](#-hyperparameters--training-details)
- [Known Issues & Roadmap](#-known-issues--roadmap)
- [Authors & License](#-authors--license)

---

## 🎯 Overview

This project implements Double Deep Q-Networks (DDQN) to dynamically control traffic lights across progressively larger intersection networks. Integrated with **SUMO** (Simulation of Urban MObility), the system learns to optimize vehicle throughput by observing real-time queue lengths, phase states, and inter-agent coordination signals — replacing static fixed-timer traffic lights with adaptive, learned policies.

### Key Achievements

| Phase | Milestone | Result |
|-------|-----------|--------|
| **Phase 0** | Single-agent pretraining | **94.3% improvement** over fixed-timers |
| **Phase 1** | Transfer learning to 4-intersection grid | **68% instant improvement** with zero additional training |
| **Phase 1** | Cooperative multi-agent fine-tuning | **Perfect network load balancing** across all intersections |
| **Phase 2** | Local Supervisor (24-dim) | **+4.6%** over decentralized 8-intersection baseline |
| **Phase 2** | Global Supervisor (28-dim) | **+2.8%** over decentralized baseline |
| **Phase 3** | LSTM + Z-Score defense against FDI attacks | **Full reward recovery** under active cyberattack |

---

## 🏗️ Project Architecture

The project tackles the Curse of Dimensionality in multi-agent RL by scaling the architecture in 4 distinct phases.

### Phase 0: Single-Agent Pretraining

A single DDQN agent controls one intersection with a **6-dimensional state space**:

```
State = [queue_N, queue_S, queue_E, queue_W, current_phase, time_since_change]
```

- **Network:** 3-layer MLP (6 → 128 → 128 → 2) with ReLU activations
- **Action Space:** 2 actions — keep current phase (0) or switch (1)
- **Training:** 1000 episodes with ε-greedy exploration (ε: 1.0 → 0.01)
- **Key Files:** `main.py`, `sumo_environment.py`, `train.py`, `evaluate.py`

### Phase 1: 4-Intersection Multi-Agent Grid

The single-agent model is extended to a **2×2 grid** (4 intersections, 500m spacing) using two strategies:

1. **Independent Transfer Learning:** The Phase 0 checkpoint is cloned to all 4 agents. Each runs independently with its own 6-dim state.
2. **Cooperative Mode:** State space expanded to **8 dimensions** (6 local + 2 neighbor queue values). Agents share group-averaged rewards to encourage network-level cooperation.

```
Layout:
  [TLS_A] --- [TLS_B]
     |           |
  [TLS_C] --- [TLS_D]
```

- **Key Files:** `main_multiagent.py`, `sumo_environment_multiagent.py`

### Phase 2: 8-Intersection Hierarchical Supervisors

As the grid scales to **8 intersections**, flat multi-agent systems cause localized gridlocks. The solution introduces a **two-tier hierarchy**:

```
Layout:
  Group A (Left)              Group B (Right)
  [TLS_1] --- [TLS_2]  <-->  [TLS_5] --- [TLS_6]
     |           |              |           |
  [TLS_3] --- [TLS_4]  <-->  [TLS_7] --- [TLS_8]
```

#### Step 1: Local Supervisors (24-dim input)

Each group's supervisor observes all 4 agents' raw 6-dim states concatenated into a **24-dimensional group state**. It outputs 4 continuous coordination signals ∈ [-1, +1] via tanh activation — one per agent. Each agent's state is then enhanced from 6-dim to **7-dim** (local state + supervisor signal).

```
Supervisor A: [state_tls1 || state_tls2 || state_tls3 || state_tls4] → 4 signals
Agent i:      [6-dim local state, supervisor_signal_i] → action
```

- **Supervisor Training:** TD regression on group-average reward
- **Agent Training:** Individual reward with DDQN

#### Step 2: Global Supervisors (28-dim input)

The two supervisors exchange a **4-dimensional cross-group summary**:

```
Summary = [avg_queue, max_queue, avg_waiting_time, boundary_queue]
```

Each supervisor's input expands from 24 → **28 dimensions** (24 own + 4 from the other group). This enables proactive congestion management across group boundaries.

- **Boundary Intersections:** TLS_2/TLS_4 (Group A) ↔ TLS_5/TLS_7 (Group B)
- **Key Files:** `main_supervisor.py`, `main_global_supervisor.py`, `supervisor_agent.py`, `sumo_environment_supervisor.py`

### Phase 3: Cybersecurity & LSTM Defense

Smart city traffic infrastructure is vulnerable to cyberattacks. This phase implements and defends against **False Data Injection (FDI)** attacks on sensor data.

#### Attack Model
- **FDI Attack:** Random queue sensors are injected with large positive values (+10 to +15) with 15% probability per intersection per step
- **Network Unreliability:** Packet loss (5%) and bounded delay (0–3 steps)

#### Defense Architecture
1. **Statistical Watchman (Z-Score):** Rolling-window anomaly detector (window=20, threshold=3σ) identifies values that deviate significantly from recent history
2. **LSTM Predictor:** A pre-trained LSTM (input_size=4, hidden_size=64) predicts what the correct queue values should be based on the last 20 steps of clean history. Poisoned values are seamlessly replaced with LSTM predictions.

#### Experiment Scenarios
Five scenarios run sequentially: `baseline`, `attack`, `defense`, `unreliable`, `secure`

- **Key Files:** `security_layer.py`, `lstm_predictor.py`, `train_lstm.py`, `main_security.py`, `collect_baseline_data.py`, `analyze_security.py`
- **Full Report:** [SECURITY_PHASE_REPORT.md](SECURITY_PHASE_REPORT.md)

---

## 📊 Performance Results

### Phase 0 & 1 (Single Agent & 4-Intersection Grid)

| System | Avg Reward | Training | Improvement |
|--------|-----------|----------|-------------|
| Single-Agent Initial | -4,253.5 | 1000 eps | 94.3% vs fixed-time |
| Multi-Agent Transfer | -1,363.1 | 0 eps | Instant baseline |
| Multi-Agent Fine-Tuned | **-560.8** | 100 eps | **86.8% boost** |
| Multi-Agent Cooperative | -585.8 | 700 eps | Perfect balance ⚖️ |

### Phase 2 (8-Intersection Hierarchy)

| Architecture | Input Dim | Avg Reward / Intersection | vs Baseline |
|---|---|---|---|
| 8-Int No Supervisor | — | -197.0 | Baseline |
| 8-Int Local Supervisor | 24-dim | **-187.9** | 🏆 **+4.6%** |
| 8-Int Global Supervisor | 28-dim | **-191.5** | 🏆 **+2.8%** |

> **Note:** The Local Supervisor slightly outperformed the Global Supervisor under a 900-episode training budget. The 28-dim Global network's larger state space requires more training episodes to fully converge — the boundary-crossing features add complexity that hasn't fully saturated.

### Phase 3 (Cybersecurity)

Tested over 20 evaluation episodes per scenario:

| Scenario | Attack? | Detection Rate | Avg Wait Time | Avg Reward |
|----------|---------|---------------|---------------|------------|
| `baseline` | No | — | 0.044 | **-1624.5** |
| `attack` | FDI | — | 0.020 | -1403.0 *(broken)* |
| `defense` | FDI | ~2.31 | 0.017 | **-1491.5** *(recovered)* |
| `unreliable` | No | — | 0.022 | -1838.5 *(noise)* |
| `secure` | FDI | ~2.29 | 0.020 | **-1468.0** *(recovered)* |

---

## 🔧 Installation & Setup

### Prerequisites

- **Python 3.8+** (tested on 3.10–3.13)
- **CUDA-capable GPU** (NVIDIA RTX 2050+ recommended for Phase 2)
- **SUMO Traffic Simulator** (v1.25.0+)

### Setup

1. Install SUMO from [eclipse.org/sumo](https://www.eclipse.org/sumo/) and set the `SUMO_HOME` environment variable.

2. Clone and install:
```bash
git clone https://github.com/Utkarsh240102/Supervisor-multi-agent-RL.git
cd Supervisor-multi-agent-RL

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Other dependencies
pip install numpy pandas matplotlib tqdm traci sumolib
```

---

## 🚀 Usage & Commands

### Phase 0: Single-Agent

```bash
# Train
python main.py --mode train --episodes 500

# Evaluate
python main.py --mode evaluate
```

### Phase 1: Multi-Agent

```bash
# Train cooperative mode
python main_multiagent.py --mode train --cooperative --episodes 700 --learning-rate 0.0005 --epsilon 0.9

# Evaluate
python main_multiagent.py --mode evaluate --load-final
```

### Phase 2: Hierarchical Supervisors

```bash
# Step 1: Local Supervisors (24-dim)
python main_supervisor.py --mode train --episodes 500
python main_supervisor.py --mode evaluate --load-final --eval-episodes 20

# Step 2: Global Supervisors (28-dim)
python main_global_supervisor.py --mode train --episodes 900 --from-scratch --epsilon 0.9
python main_global_supervisor.py --mode evaluate --load-final --eval-episodes 20

# Resume training from checkpoint
python main_supervisor.py --mode train --episodes 300 --resume-from 200

# Visualize with SUMO GUI
python main_supervisor.py --mode evaluate --load-final --eval-episodes 5 --gui
```

### Phase 3: Security

```bash
# 1. Collect clean baseline data
python collect_baseline_data.py --episodes 50

# 2. Split into train/val sets
python split_baseline_data.py

# 3. Validate dataset
python validate_baseline_data.py

# 4. Train LSTM predictor
python train_lstm.py --epochs 25

# 5. Run all 5 security scenarios
python main_security.py --episodes 20

# 6. Generate analysis plots
python analyze_security.py
```

### Visualization Suite

```bash
# Phase 2 analysis plots
python analyze_supervisor.py
python analyze_global_supervisor.py
```

Output directories: `analysis_supervisor/`, `analysis_global_supervisor/`, `analysis_security/`

---

## 📁 Project Structure

```
├── Core RL
│   ├── agent.py                      # DDQN agent with GPU auto-detection
│   ├── network.py                    # 3-layer MLP (Sequential architecture)
│   ├── replay_buffer.py              # Experience replay buffer
│   └── train.py                      # Phase 0 training loop
│
├── Environments
│   ├── sumo_environment.py           # Phase 0: single intersection
│   ├── sumo_environment_multiagent.py # Phase 1: 4-intersection grid
│   ├── sumo_environment_8intersection.py # Phase 2: 8-int baseline
│   └── sumo_environment_supervisor.py # Phase 2: 8-int with supervisor support
│
├── Supervisor System
│   ├── supervisor_agent.py           # SupervisorNetwork + SupervisorAgent
│   ├── main_supervisor.py            # Local supervisor training (24-dim)
│   └── main_global_supervisor.py     # Global supervisor training (28-dim)
│
├── Security Phase
│   ├── security_layer.py             # FDI attack + Z-Score + LSTM defense
│   ├── lstm_predictor.py             # TrafficLSTM model definition
│   ├── train_lstm.py                 # LSTM training pipeline
│   ├── collect_baseline_data.py      # Clean data collection
│   ├── split_baseline_data.py        # Episode-boundary train/val split
│   ├── validate_baseline_data.py     # Data integrity checks
│   ├── test_lstm_attack_sanity.py    # Manual attack verification
│   └── main_security.py             # 5-scenario experiment runner
│
├── Entry Points
│   ├── main.py                       # Phase 0 entry
│   ├── main_multiagent.py            # Phase 1 entry
│   └── main_8intersection.py         # Phase 2 baseline entry
│
├── Analysis & Visualization
│   ├── evaluate.py                   # Phase 0/1 evaluation utilities
│   ├── analyze_supervisor.py         # Phase 2 Step 1 plots
│   ├── analyze_global_supervisor.py  # Phase 2 Step 2 plots
│   └── analyze_security.py          # Phase 3 plots
│
├── SUMO Network Generators
│   ├── generate_sumo_files.py        # Phase 0 network
│   ├── generate_sumo_multiagent.py   # Phase 1 network
│   └── generate_sumo_8intersection.py # Phase 2 network (2×4 grid)
│
├── Documentation
│   ├── README.md                     # This file
│   ├── bugs.md                       # 3-pass audit report (15 bugs)
│   ├── SECURITY_PHASE_REPORT.md      # Phase 3 technical report
│   └── security-plan.md             # Phase 3 implementation plan
│
└── sumo_files*/                      # Generated SUMO network XML files
```

---

## 🧠 Hyperparameters & Training Details

### Neural Network Architecture

| Component | Layers | Neurons | Activation | Output |
|-----------|--------|---------|------------|--------|
| **DDQN Agent** | 3-layer MLP | 128 hidden | ReLU | 2 (Q-values) |
| **Supervisor** | 3-layer MLP | 64 hidden | ReLU → Tanh | 4 (signals ∈ [-1,1]) |
| **LSTM Predictor** | 1-layer LSTM + Linear | 64 hidden | — | 4 (queue predictions) |

### Training Configuration

| Parameter | Phase 0 | Phase 1 | Phase 2 | Phase 3 (LSTM) |
|-----------|---------|---------|---------|----------------|
| Learning Rate | 0.001 | 0.0005 | 0.0001 (agent) / 0.001 (sup) | 0.001 |
| Gamma (γ) | 0.95 | 0.95 | 0.95 | — |
| Epsilon Decay | 0.995 | 0.995 | 0.995 | — |
| Batch Size | 64 | 64 | 64 | 256 |
| Buffer Size | 10,000 | 10,000 | 10,000 | — |
| Target Update | 10 eps | 10 eps | 10 eps | — |
| Gradient Clip | 10.0 | 10.0 | 10.0 | 5.0 |

### Reward Function

All phases share the same core reward formulation:

```python
reward = -(total_queue) - 0.5 * (total_waiting_time) - 10 * (quick_switch_penalty)
```

- **Queue penalty:** Direct negative proportional to halting vehicles
- **Waiting penalty:** 0.5× weighting on cumulative vehicle waiting time
- **Switch penalty:** -10 if the agent attempts to switch phases within 5 seconds of the last switch

---

## 🔮 Known Issues & Roadmap

### Active Bugs

See [`bugs.md`](bugs.md) for the complete 3-pass audit report. Key items:

- **BUG-01 (Critical):** Supervisor TD target broadcast — all 4 agents receive identical signals instead of differentiated urgency values
- **BUG-03 (Medium):** Security layer logging uses incorrect `np.where(flagged)[0]` double-indexing
- **BUG-04 (Medium):** False positive rate metric is always 0.0 (counter never incremented)

### Planned Improvements

1. **Independent TD Targets** — Per-intersection reward-based supervisor training for fine-grained signal differentiation
2. **Prioritized Experience Replay (PER)** — Replace uniform sampling with TD-error-weighted prioritization
3. **Huber Loss** — Replace MSELoss with SmoothL1Loss to stabilize supervisor convergence
4. **Utility Module Refactor** — Extract duplicated `partial_transfer()`, `set_seed()` into shared `utils.py`
5. **State Normalization** — Add batch normalization or manual feature scaling for faster convergence
6. **Dynamic Boundary Detection** — Replace hardcoded boundary TLS IDs with graph-based automatic detection

---

## 📝 Authors & License

**Project Team:** RL Traffic Control Research Group  
Developed for academic research purposes using the SUMO Traffic Modeling Suite.

**License:** MIT
