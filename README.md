# 🚦 Hierarchical Multi-Agent RL Traffic Control System

**Advanced Deep Reinforcement Learning for Large-Scale Traffic Coordination**

> A scalable, hierarchical multi-agent reinforcement learning system that implements cluster-based supervisors to manage dynamic traffic flow across an 8-intersection urban grid. This project demonstrates how introducing a centralized supervisor layer on top of decentralized local agents significantly reduces network congestion.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [System Architecture (The Hierarchy)](#-system-architecture-the-hierarchy)
- [Key Features](#-key-features)
- [Performance Results](#-performance-results)
- [Installation](#-installation)
- [Usage (Training & Evaluation)](#-usage-training--evaluation)
- [Visualizations & Analysis](#-visualizations--analysis)

---

## 🎯 Overview

As traffic networks scale, flat multi-agent systems (where every traffic light is an independent agent) struggle to coordinate flow, leading to localized gridlocks. 

This project solves that by introducing a **Hierarchical Supervisor Architecture** built in PyTorch and integrated with the SUMO (Simulation of Urban MObility) engine. 

### Key Achievements
- ✅ **Tested on an 8-Intersection Grid** (a complex rectangular urban network).
- ✅ **Successfully implemented Local Supervisors** that manage clusters of 4 intersections, yielding a **+4.6% improvement** in throughput vs the decentralized baseline.
- ✅ **Successfully implemented Global Supervisor cross-talk**, allowing cluster managers to exchange 4-dimensional summaries to prevent feeding traffic into jammed neighboring zones.
- ✅ **Completely automated visualization suite** for tracking neural network convergence and comparing architectures.

---

## 🏗️ System Architecture (The Hierarchy)

Our architecture is split into two distinct tiers of Intelligence:

### 1. The Local Agents (Tier 1)
There are **8 Deep Q-Network (DDQN) Agents**, one for each traffic light. 
- **State Input (7-dim):** 6 local features (queues, phases) + 1 explicit Urgency Signal from their respective Supervisor.
- **Action:** Choose to extend the current traffic phase or switch to the next one.
- **Goal:** Clear its immediate intersection.

### 2. The Supervisors (Tier 2)
The 8 intersections are split into **Group A (tls 1-4)** and **Group B (tls 5-8)**. Each group has a single Supervisor model.

#### Phase 1: Local Supervisor (Blind Clusters)
- The supervisor reads the exact states of all 4 agents in its group (**24-dimensional input**).
- It calculates an urgency signal in the range `[-1, 1]` for each agent.
- It is trained on the *average group reward*, forcing it to optimize the cluster's overall health rather than selfish individual agents.

#### Phase 2: Global Supervisor (Connected Clusters)
- Supervisors compute a `[avg_queue, max_queue, avg_wait_time, boundary_queue]` summary of their cluster.
- They exchange this summary globally.
- The supervisor's state space expands to **28-dimensions** (24 local + 4 neighbor summary) so it can preemptively react to traffic jams in the adjacent block.

---

## 📊 Performance Results

All models were evaluated rigorously over 20 randomized episodes using `--random` spawning in SUMO. 

| Traffic System Model | Architecture Complexity | Avg Reward / Intersection | Improvement vs Baseline |
| :--- | :--- | :--- | :--- |
| **8-Int Baseline** | Decentralized DDQN neighbors | -197.0 | `Baseline` |
| **Local Supervisor** | 24-dim distinct clusters | **-187.9** | 🏆 **+4.6%** |
| **Global Supervisor**| 28-dim connected clusters | **-191.5** | **+2.8%** |

***Note on Dimensionality:** Under our 900-episode training limit, the purely focused *Local Supervisor* marginally outperformed the *Global Supervisor*. The Global model's 28-dimensional state space induced slight hesitation at boundary intersections. However, *both* hierarchical models conclusively beat the flat multi-agent baseline.*

---

## 🔧 Installation

### Prerequisites
- **Python 3.8+** 
- **CUDA-capable GPU** (highly recommended for training multi-agent DDQN)
- **SUMO Traffic Simulator** (Version 1.25.0+)

### Setup
Ensure SUMO is installed properly and system environment variables (`SUMO_HOME`) are configured.
```bash
# Clone the repository
git clone https://github.com/yourusername/RL-project-Multiagent_superviser.git
cd RL-project-Multiagent_superviser

# Create a virtual environment and install requirements
python -m venv .venv
source .venv/bin/activate  # (or .venv\Scripts\activate on Windows)
pip install torch numpy pandas matplotlib tqdm traci sumolib
```

---

## 🚀 Usage (Training & Evaluation)

### Step 1: Local Supervisor Mode 
Run the purely cluster-based supervisors (24-dim neural network).

**To Train from scratch (~500 episodes):**
```bash
python main_supervisor.py --mode train --episodes 500
```
**To Evaluate final models visually (with SUMO GUI):**
```bash
python main_supervisor.py --mode evaluate --load-final --eval-episodes 5 --gui
```

### Step 2: Global Supervisor Mode 
Run the cross-communicating supervisor network (28-dim neural network).

**To Train from scratch (~900 episodes):**
```bash
python main_global_supervisor.py --mode train --episodes 900 --from-scratch --epsilon 0.9
```
**To Evaluate final models:**
```bash
python main_global_supervisor.py --mode evaluate --load-final --eval-episodes 20
```

---

## 📈 Visualizations & Analysis

The project comes with dedicated visualization scripts to immediately graph the RL training convergence, per-intersection performance breaksdowns, and full system comparative analysis.

**To analyze the Local Supervisor:**
```bash
python analyze_supervisor.py
# Outputs to: analysis_supervisor/
```

**To analyze the Global Supervisor:**
```bash
python analyze_global_supervisor.py
# Outputs to: analysis_global_supervisor/
```
