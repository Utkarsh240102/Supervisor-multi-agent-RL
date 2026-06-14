# 🚦 Hierarchical Multi-Agent DDQN Traffic Light Control System

**Deep Reinforcement Learning for Scalable, Secure Urban Traffic Management**

> A multi-phase research project that evolves automated traffic control from a single isolated intersection to a **hierarchically coordinated 8-intersection urban network** with **cyberattack-resilient LSTM-defended sensors** — built on SUMO, PyTorch, and DDQN.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Performance & Visual Summary](#-performance--visual-summary)
- [Project Architecture](#-project-architecture)
  - [Phase 0: Single-Agent Pretraining](#phase-0-single-agent-pretraining)
  - [Phase 1: 4-Intersection Multi-Agent Grid](#phase-1-4-intersection-multi-agent-grid)
  - [Phase 2: 8-Intersection Hierarchical Supervisors](#phase-2-8-intersection-hierarchical-supervisors)
  - [Phase 3: Cybersecurity & LSTM Defense](#phase-3-cybersecurity--lstm-defense)
- [Installation & Setup](#-installation--setup)
- [Usage & Commands](#-usage--commands)
- [Project Structure](#-project-structure)
- [Future Roadmap](#-future-roadmap)
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

## 📊 Performance & Visual Summary

Our comprehensive multi-phase analysis proves the efficacy of the hierarchical approach. Below is the cross-phase performance summary demonstrating nearly **+97% improvement** over baselines at maximum scale:

![Cross-Phase Performance Summary](figures/fig13_cross_phase_summary.png)

---

## 🏗️ Project Architecture

The project tackles the Curse of Dimensionality in multi-agent RL by scaling the architecture in 4 distinct phases.

### Phase 0: Single-Agent Pretraining

**The Concept:** Before tackling a massive grid, we must first prove that an RL agent can learn the foundational physics of traffic control (e.g., clearing queues, avoiding rapid flickering). Training a single isolated agent creates a clean, noise-free environment to establish a robust baseline policy.

A single DDQN agent controls one intersection with a **6-dimensional state space**:

```
State = [queue_N, queue_S, queue_E, queue_W, current_phase, time_since_change]
```

- **Network:** 3-layer MLP (6 → 128 → 128 → 2) with ReLU activations
- **Action Space:** 2 actions — keep current phase (0) or switch (1)
- **Training:** 1000 episodes with ε-greedy exploration (ε: 1.0 → 0.01)

#### Phase 0 Results:
![Phase 0 Training Curve](figures/fig2_phase0_training_curve.png)
<details>
<summary>View Additional Diagnostics</summary>
<br>
<img src="figures/fig7_phase0_training_diagnostics.png" alt="Diagnostics" width="45%">
<img src="figures/fig8_phase0_wait_queue.png" alt="Wait Queue" width="45%">
</details>

### Phase 1: 4-Intersection Multi-Agent Grid

**The Concept:** Scaling to a 2×2 grid introduces the "selfish agent" problem, where one intersection might clear its own queue simply by dumping traffic into its neighbor. We solve this by bridging the agents together.

The single-agent model is extended to a **2×2 grid** (4 intersections, 500m spacing) using two strategies:

1. **Independent Transfer Learning:** Rather than training from scratch, the Phase 0 checkpoint is cloned to all 4 agents. This jumpstarts training by transferring foundational traffic knowledge instantly.
2. **Cooperative Mode:** State space expanded to **8 dimensions** (6 local + 2 neighbor queue values). Agents share group-averaged rewards, forcing them to cooperate and balance network-level load.

#### Phase 1 Results:
![Transfer Fine-Tuning](figures/fig3_phase1_transfer_finetuning.png)
![Cooperative vs Independent](figures/fig10_phase1_coop_vs_independent.png)
<details>
<summary>View Individual Agent Performance</summary>
<br>
<img src="figures/fig9_phase1_per_agent.png" alt="Per Agent">
</details>


### Phase 2: 8-Intersection Hierarchical Supervisors

**The Concept:** As the grid scales to 8 intersections, flat multi-agent systems suffer from the Curse of Dimensionality—local agents cannot see the "big picture" of incoming traffic waves. We solve this by introducing regional traffic managers.

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

#### Step 2: Global Supervisors (28-dim input)
The two supervisors exchange a **4-dimensional cross-group summary** enabling proactive congestion management across boundaries.

#### Phase 2 Results:
![Hierarchical Convergence](figures/fig4_phase2_hierarchical_convergence.png)
<details>
<summary>View Supervisor Loss & Agent Diagnostics</summary>
<br>
<img src="figures/fig12_supervisor_loss.png" alt="Supervisor Loss" width="45%">
<br>
<img src="figures/fig11_phase2_per_agent_8.png" alt="Per Agent 8" width="80%">
</details>


### Phase 3: Cybersecurity & LSTM Defense

**The Concept:** Real-world IoT sensors break or get hacked. If a sensor falsely reports a massive traffic jam (False Data Injection), the RL agent will prioritize an empty road, causing immediate gridlock. We build an AI "immune system" to detect and filter out these lies.

Smart city traffic infrastructure is vulnerable to cyberattacks. This phase implements and defends against **False Data Injection (FDI)** attacks on sensor data.

#### Defense Architecture
1. **Statistical Watchman (Z-Score):** Rolling-window anomaly detector (window=20, threshold=3σ) identifies values that deviate significantly from recent history.
2. **LSTM Predictor:** A pre-trained LSTM (input_size=4, hidden_size=64) predicts what the correct queue values should be based on the last 20 steps of clean history. Poisoned values are seamlessly replaced.

#### Phase 3 Results:
![Scenario Comparison](figures/fig5_security_scenario_comparison.png)

The radar chart below highlights the multi-metric success of the defense mechanism, recovering the system entirely from the attack state back to baseline performance:
![Security Radar](figures/fig14_security_radar.png)

And here is a live view of the LSTM actively correcting a malicious FDI spike:
![LSTM Attack Correction](figures/fig6_lstm_attack_correction.png)

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
python main.py --mode train --episodes 500
python main.py --mode evaluate
```

### Phase 1: Multi-Agent
```bash
python main_multiagent.py --mode train --cooperative --episodes 700 --learning-rate 0.0005 --epsilon 0.9
python main_multiagent.py --mode evaluate --load-final
```

### Phase 2: Hierarchical Supervisors
```bash
python main_supervisor.py --mode train --episodes 500
python main_global_supervisor.py --mode train --episodes 900 --from-scratch --epsilon 0.9
```

### Phase 3: Security
```bash
python collect_baseline_data.py --episodes 50
python train_lstm.py --epochs 25
python main_security.py --episodes 20
```

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
│   └── main_security.py             # 5-scenario experiment runner
│
├── Entry Points
│   ├── main.py                       # Phase 0 entry
│   ├── main_multiagent.py            # Phase 1 entry
│   └── main_8intersection.py         # Phase 2 baseline entry
│
├── Analysis & Visualization
│   ├── generate_paper_figures.py     # High-quality paper plotting
│   └── generate_extra_figures.py     # Extended diagnostics plotting
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

## 🔮 Future Roadmap

1. **Prioritized Experience Replay (PER)** — Replace uniform sampling with TD-error-weighted prioritization
2. **Huber Loss** — Replace MSELoss with SmoothL1Loss to stabilize supervisor convergence
3. **State Normalization** — Add batch normalization or manual feature scaling for faster convergence
4. **Dynamic Boundary Detection** — Replace hardcoded boundary TLS IDs with graph-based automatic detection

---

## 📝 Authors & License

**Project Team:** RL Traffic Control Research Group  
Developed for academic research purposes using the SUMO Traffic Modeling Suite.

**License:** MIT
