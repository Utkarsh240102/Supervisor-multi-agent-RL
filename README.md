# 🚦 Multi-Agent DDQN Traffic Light Control System

**Advanced Deep Reinforcement Learning for Intelligent Traffic Management**

> A highly scalable multi-agent reinforcement learning ecosystem. This project charts the entire progression from a single-intersection DDQN controller to a massive, hierarchically-coordinated 8-intersection urban grid. It demonstrates transfer learning, cooperative neighbor polling, and centralized cluster management architecture to drastically reduce traffic gridlock.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Phase 0 & 1: Single Agent & Flat Multi-Agent (2x2 Grid)](#-phase-0--1-single-agent--flat-multi-agent-2x2-grid)
- [Phase 2: Hierarchical Supervisors (8-Intersection Grid)](#-phase-2-hierarchical-supervisors-8-intersection-grid)
- [Comprehensive Performance Results](#-comprehensive-performance-results)
- [Installation](#-installation)
- [Usage & Commands](#-usage--commands)
- [Project Architecture](#-project-architecture)

---

## 🎯 Overview

This repository uses **PyTorch** and **SUMO** (Simulation of Urban MObility) to train autonomous traffic lights. By replacing fixed-timer traffic lights with adaptive neural networks (DDQN), the system dynamically reacts to real-time traffic queues and waiting times to clear congestion beautifully.

We approached this problem in distinct scaling phases:
1. **Phase 0:** Master a single, isolated intersection.
2. **Phase 1:** Scale up to a 2x2 grid (4 intersections) using transfer learning and basic neighbor awareness.
3. **Phase 2:** Scale up to an 8-intersection grid and introduce a centralized "Supervisor" neural network to manage clusters.

---

## 📈 Phase 0 & 1: Single Agent & Flat Multi-Agent (2x2 Grid)

### Phase 0: Single Agent Pretraining
We began by training a standard Double Deep Q-Network (DDQN) explicitly on a single intersection. 
- **Achievement:** Improved average reward by **+61%** over 500 episodes and completely dominated fixed-timer traffic lights by +94.3%. Waiting time plummeted from 0.73s to 0.38s.

### Phase 1: 4-Intersection Transfer Learning
We expanded the map to a `2x2` grid (4 intersections) and transferred the brain of our single-agent to control all 4 simultaneously. 
- **Independent Mode (-560.8 avg reward):** The agents ran independently with no communication. Fine-tuning for just 100 episodes boosted performance radically compared to raw transfer learning.
- **Cooperative Mode (-585.8 avg reward):** Agents were updated to "look" at their neighbor's queue. While total throughput remained similar, they achieved **perfect load balancing** across the network, ending localized jams.

---

## 👑 Phase 2: Hierarchical Supervisors (8-Intersection Grid)

As the network expanded to a heavy 8-intersection layout, "flat" multi-agent systems struggled to maintain a cohesive flow. We solved this by splitting the grid into **Group A (tls 1-4)** and **Group B (tls 5-8)** and placing a powerful Supervisor Neural Network above each cluster.

### Step 1: Local Supervisors
The localized supervisor digests all 4 agents' real-time stats simultaneously (**24-dimensional view**) and dispatches a coordinating 'urgency signal' to govern the entire subset. 
- **Achievement:** This cluster-aware behavior crushed the flat baseline by **+4.6%**, smoothing out flow inside the groups.

### Step 2: Global Supervisors (Cross-Talk)
We enabled Supervisor A and Supervisor B to talk. They continuously compute a 4-dimensional macroeconomic summary (`[avg_queue, max_queue, avg_time, boundary_queue]`) and share it with each other. The neural-net dimensions expanded to **28 input features**.
- **Achievement:** The network officially gained cross-border awareness. It beats the baseline by **+2.8%** and proactively halts traffic bound toward jammed neighboring clusters.

---

## 📊 Comprehensive Performance Results

Here is the ultimate scorecard across all tested multi-agent architectures (per intersection avg):

| Generation | System Architecture | Avg Reward / Intersection | Improvement Profile |
| :--- | :--- | :--- | :--- |
| **Phase 1** | 4-Int Cooperative | -585.8 *(congested baseline)*| Perfect node balancing |
| **Phase 2 Baseline** | 8-Int Flat Multi-Agent | -197.0 | New standard load |
| **Phase 2 Step 1** | **8-Int Local Supervisor** | **-187.9** | 🏆 **+4.6% vs Baseline** |
| **Phase 2 Step 2** | **8-Int Global Supervisor** | **-191.5** | 🏆 **+2.8% vs Baseline** |

*(Note on Dimensionality: In Phase 2 under our 900-episode cap, the highly-focused 24-dim Local Supervisor technically edged out the 28-dim Global Supervisor due to the "Curse of Dimensionality" causing slight hesitation at traffic borders. Both decisively defeated the baseline.)*

---

## 🔧 Installation

### Prerequisites
- **Python 3.8+** (Tested on Python 3.10-3.13)
- **CUDA-capable GPU** (Recommended: NVIDIA RTX 2050 or better)
- **SUMO Traffic Simulator** (Version 1.25.0+)

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/RL-project-Multiagent_superviser.git
cd RL-project-Multiagent_superviser

# Create a virtual environment and load dependencies
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install torch numpy pandas matplotlib tqdm traci sumolib
```

---

## 🚀 Usage & Commands

### Running Phase 1 (Single / 4-Intersection)
To run the original independent or cooperative grid simulations from Phase 0/1, refer to the legacy runner scripts located in the root (e.g. `main.py`, `main_cooperative.py`). 

### Running Phase 2 (8-Intersection Supervisor)

**1. To run the Step 1 Local Supervisors (-187.9 score):**
```bash
# Train from scratch (500 episodes)
python main_supervisor.py --mode train --episodes 500

# Evaluate visually inside SUMO
python main_supervisor.py --mode evaluate --load-final --eval-episodes 5 --gui
```

**2. To run the Step 2 Global Supervisors (-191.5 score):**
```bash
# Train from scratch (900 episodes)
python main_global_supervisor.py --mode train --episodes 900 --from-scratch --epsilon 0.9

# Evaluate headless purely for metrics
python main_global_supervisor.py --mode evaluate --load-final --eval-episodes 20
```

### Visualizing Results
We built customized Python plotting suites to trace the Neural Network TD losses and episode stability.
To view learning curves, run:
```bash
python analyze_supervisor.py
python analyze_global_supervisor.py
```
*Outputs will be saved dynamically as `.png` files to the `/analysis_x` directories.*

---

## 🧠 Project Architecture Details

### Neural Definitions
- **Local DDQN Architecture:** 2 Hidden layers (128 units), ReLU activation.
- **Supervisor Architecture:** 2 Hidden layers (64 units), Tanh activation for urgent continuous limits `[-1, 1]`.
- **Epsilon Greedy:** Linear decay ensuring early exploration of all traffic phases.

### Reward Function Equation
```python
reward = -(queue_length + 0.5 * waiting_time + 10 * phase_switch_penalty)
```
*Punishes severe build-ups explicitly while discouraging hyper-frequent light flicking.*
