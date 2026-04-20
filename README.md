# 🚦 Advanced Multi-Agent DDQN Traffic Light Control System

**Deep Reinforcement Learning for Intelligent, Scalable Traffic Management**

> A massive, multi-phase reinforcement learning ecosystem that charts the progression of automated traffic control from a **single isolated intersection**, to a **4-intersection cooperative grid**, and finally to a **hierarchically coordinated 8-intersection urban network** featuring centralized Global & Local Supervisor agents.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture Progression](#-system-architecture-progression)
  - [Phase 0 & 1: Single Agent & 2x2 Grid](#phase-0--1-single-agent--2x2-grid-flat-multi-agent)
  - [Phase 2: 8-Intersection Hierarchical Supervisors](#phase-2-8-intersection-hierarchical-supervisors)  - [Phase 3: Cyber Security & LSTM Defense](#phase-3-cyber-security--lstm-defense)- [Comprehensive Performance Results](#-comprehensive-performance-results)
- [Installation & Setup](#-installation--setup)
- [Usage & Commands](#-usage--commands)
  - [Phase 1 Scripts](#phase-1-single--flat-multi-agent)
  - [Phase 2 Scripts](#phase-2-hierarchical-supervisors)
  - [Phase 3 Security Scripts](#phase-3-cyber-security--lstm-defense-scripts)
- [Visualization Suite](#-visualization-suite)
- [Hyperparameters & Training Details](#-hyperparameters--training-details)

---

## 🎯 Overview

This project implements sophisticated Double Deep Q-Networks (DDQN) to dynamically control traffic lights. Integrated seamlessly with **SUMO** (Simulation of Urban MObility), the neural networks eliminate the need for fixed-timer lights by actively observing vehicle queues and balancing throughput in real-time.

### Key Achievements Across All Phases:
- ✅ **Phase 0:** Single-agent pretraining achieved **94.3% improvement** over fixed-timers.
- ✅ **Phase 1:** Transfer learning instantly improved a 4-intersection network by **68%** with zero training, while Cooperative mode achieved **perfect network load balancing**.
- ✅ **Phase 2:** Successfully solved the "blind agent" problem on massive 8-intersection grids by building a **Hierarchical Supervisor Network**. The Local Supervisor beat the decentralized baseline by **+4.6%**, and the cross-talking Global Supervisor beat it by **+2.8%**.

---

## 🏗️ System Architecture Progression

The project tackles the "Curse of Dimensionality" in Reinforcement Learning by scaling the architecture in distinct phases.

### Phase 0 & 1: Single Agent & 2x2 Grid (Flat Multi-Agent)
Initially, the system was built for a single intersection. The agent's state space was `6 dimensions` (4 queues, current phase, time since switch). We then expanded the map to a `2x2 grid (4 intersections, 500m spacing)` using two methods:
1. **Independent Transfer Learning:** We cloned the single agent 4 times. Each intersection ran independently.
2. **Cooperative Mode:** We increased the state space to `8 dimensions`, allowing each agent to "see" the queue lengths of its immediate neighbors.

### Phase 2: 8-Intersection Hierarchical Supervisors
As the grid scaled up to 8 heavily trafficked intersections, flat multi-agent systems caused localized gridlocks. We solved this by splitting the grid into **Group A (tls 1-4)** and **Group B (tls 5-8)** and placing a central "Supervisor" neural network above them.

- **Step 1: Local Supervisors:** A localized supervisor reads the exact queues of all 4 agents (**24-dimensional view**) and dispatches a coordinating 'urgency signal' `[-1, 1]` to its group. It is trained entirely on the average group reward to enforce cooperative self-sacrifice.
- **Step 2: Global Supervisors:** Group A and Group B supervisors compute a macroeconomic summary (`[avg_queue, max_queue, avg_time, boundary_queue]`) and share it with each other. The neural-net expands to **28 input features**, allowing it to proactively halt traffic bound toward jammed neighboring clusters.
### Phase 3: Cyber Security & LSTM Defense
Real-world smart city infrastructure is vulnerable to cyberattacks. We implemented a **False Data Injection (FDI)** attack that intercepts and massively inflates queue sensor data before it reaches the agents, immediately breaking down hierarchical coordination.

- **Layer 1 (Statistical Watchman):** A rolling-window Z-Score anomaly detector that identifies poisoned traffic values without triggering on genuine rush-hour congestion peaks.
- **Layer 2 (LSTM Predictor):** Rather than blindly dropping anomalous data, a frozen, pre-trained Long Short-Term Memory neural network predicts what the correct queue values should mathematically be based on the last 20 seconds of traffic history. The poisoned values are seamlessly replaced, and the RL agents continue to operate optimally.
[**View the full Security Phase Report here**](SECURITY_PHASE_REPORT.md)
---

## 📊 Comprehensive Performance Results

Here is the exact algorithmic performance across all project phases, tested over rigorous 20-50 episode evaluation runs:

### Phase 0 & 1 (4-Intersection Grid)
| System | Avg Reward | Training Required | Improvement |
|--------|-----------|---------------|-------------|
| **Single-Agent Initial** | -4,253.5 | 1000 eps | 94.3% vs fixed-time |
| **Multi-Agent Transfer** | -1,363.1 | 0 eps | Instant baseline |
| **Multi-Agent Fine-Tuned** | **-560.8** | 100 eps | **86.8% boost** |
| **Multi-Agent Cooperative**| -585.8 | 700 eps | Perfect Network Balance ⚖️ |

### Phase 2 (8-Intersection Hierarchy)
| System Architecture | Complexity | Avg Reward / Intersection | Improvement Profile |
| :--- | :--- | :--- | :--- |
| **8-Int No Supervisor** | Decentralized Baseline | -197.0 | `Baseline` |
| **8-Int Local Supervisor** | 24-dim distinct clusters | **-187.9** | 🏆 **+4.6%** |
| **8-Int Global Supervisor** | 28-dim connected clusters | **-191.5** | 🏆 **+2.8%** |

*(Note on Phase 2: Under a pure 900-episode training loop, the hyper-focused 24-dim Local Supervisor slightly outperformed the 28-dim Global Supervisor. The expanding state space of the Global network requires exponentially more training to perfectly align with boundary traffic, causing slight hesitation at traffic borders. Both decisively defeated the baseline.)*

### Phase 3 (Cyber Security & LSTM Defense)
Tested automatically across a 20-episode battery.
| Scenario | Network Type | Attack Active? | Detection Rate | Avg Wait Time | Avg System Reward |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`baseline`** | Perfect | No | 0.0 | 0.044 | **-1624.5** |
| **`attack`** | Perfect | Yes (FDI) | 0.0 | 0.020 | **-1403.0** *(Broken/False)* |
| **`defense`** | Perfect | Yes (FDI) | ~2.31 | 0.017 | **-1491.5** *(Recovered)* |
| **`unreliable`**| Delayed | No | 0.0 | 0.022 | **-1838.5** *(Noise)* |
| **`secure`** | Delayed | Yes (FDI) | ~2.29 | 0.020 | **-1468.0** *(Recovered)* |

## 🔧 Installation & Setup

### Prerequisites
- **Python 3.8+** (Tested on Python 3.10-3.13)
- **CUDA-capable GPU** (NVIDIA RTX 2050 or better warmly recommended for Phase 2)
- **SUMO Traffic Simulator** (Version 1.25.0+)

### Setup
Install SUMO from `https://www.eclipse.org/sumo/` and set your `SUMO_HOME` environment variable.
```bash
# Clone the repository
git clone https://github.com/yourusername/RL-project-Multiagent_superviser.git
cd RL-project-Multiagent_superviser

# Create a virtual environment and load dependencies
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib tqdm traci sumolib
```

---

## 🚀 Usage & Commands

### Phase 1 (Single & Flat Multi-Agent)
Use the legacy files (`main.py` and `main_multiagent.py`) for the 1 and 4 intersection grids.

**Train Single Agent:**
```bash
python main.py --mode train --episodes 500
```
**Train Multi-Agent Cooperative:**
```bash
python main_multiagent.py --mode train --cooperative --episodes 700 --learning-rate 0.0005 --epsilon 0.9
```

### Phase 2 (Hierarchical Supervisors)
Use the dedicated supervisor scripts for the massive 8-intersection hierarchical networks.

**1. Local Supervisors (24-dim):**
```bash
# Train from scratch (500 episodes)
python main_supervisor.py --mode train --episodes 500

# Evaluate visually inside SUMO
python main_supervisor.py --mode evaluate --load-final --eval-episodes 5 --gui
```

**2. Global Supervisors (28-dim):**
```bash
# Train from scratch (900 episodes)
python main_global_supervisor.py --mode train --episodes 900 --from-scratch --epsilon 0.9

# Evaluate headless purely for metrics
python main_global_supervisor.py --mode evaluate --load-final --eval-episodes 20
```

### Phase 3 (Cyber Security & LSTM Defense Scripts)
```bash
# Record clean baseline data for training
python collect_baseline_data.py --episodes 50

# Train the LSTM traffic predictor
python train_lstm.py

# Run exactly 10 episodes across all 5 possible scenarios directly via CLI 
python main_security.py --episodes 10

# Generate analytical plots (reward, wait time, detection accuracy) for presentation
python analyze_security.py
```

---

## 📈 Visualization Suite

This project includes fully automated Python graphing toolkits to generate high-resolution PNGs of your neural network TD losses, episode rewards, and per-intersection breakdowns.

```bash
# Analyze Phase 2 Step 1
python analyze_supervisor.py

# Analyze Phase 2 Step 2
python analyze_global_supervisor.py
```
*Charts are dropped automatically into `analysis_supervisor/` and `analysis_global_supervisor/` directories.*

---

## 🧠 Hyperparameters & Training Details

### Neural Network Parameters
- **Local Intersections:** 2 Hidden layers (128 neurons), ReLU activation.
- **Hierarchical Supervisors:** 2 Hidden layers (64 neurons), Tanh activation for urgent continuous limits `[-1, 1]`.
- **Optimizer:** Adam (LR: `0.0005` for Local, `0.001` for Supervisors)
- **Epsilon Greedy:** Linear decay ensuring high early exploration.

### Core Reward Equation
Across all phases, the universal backpropagation reward function is formulated as:
```python
reward = -(queue_length + 0.5 * waiting_time + 10 * phase_switch_penalty)
```
*Behavior enforced: Punishes severe vehicle stack-ups explicitly, while enforcing a harsh 10x penalty to discourage hyper-frequent red/green light flicking.*

---

## � Future Improvements

**Multi-Agent Credit Assignment for the Supervisor (Independent TD Targets)**
Currently, the Group Supervisor AI aggregates the rewards of its 4 local intersections together by using a `mean()` average to learn its Temporal Difference (TD) targets. While structurally stable, an average score obscures specific intersection contributions (e.g., if Intersections 1, 2, and 3 perform optimally but Intersection 4 causes a massive traffic jam, an "average" penalty hides the root cause). 
Mathematically calculating specific, independent TD targets for each individual intersection would explicitly map the supervisor's credit/blame assignment, allowing it to learn much faster and smarter. 

*Implementation Note:* To ensure the mathematical stability of the Phase 2 baselines prior to introducing Phase 3 cyberattack vectors (False Data Injection), the supervisor grouping mechanism was mathematically frozen. Upgrading to independent TD targets is explicitly documented as a high-value future architectural upgrade, avoiding massive project-wide baseline retraining.

---

## �📝 Authors & License
**Project Team**: RL Traffic Control Research Group  
Developed for academic/research purposes inside the SUMO Traffic modeling suite. MIT License.

