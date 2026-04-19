"""
Security-phase experiment runner.

Step 6.1 scope:
- load_trained_system(env): load 8 DDQN agents + 2 supervisors from final checkpoints
- run_single_scenario(...): evaluate one scenario loop with security layer inserted

This file does not modify existing training/evaluation files.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np

from agent import DDQNAgent
from security_layer import SecurityLayer
from sumo_environment_supervisor import SupervisorSumoEnvironment
from supervisor_agent import SupervisorAgent


CHECKPOINT_DIR_AGENTS = "checkpoints_supervisor/agents"
CHECKPOINT_DIR_SUPERVISORS = "checkpoints_supervisor/supervisors"


def load_trained_system(
    env: SupervisorSumoEnvironment,
) -> Tuple[Dict[str, DDQNAgent], SupervisorAgent, SupervisorAgent]:
    """
    Load frozen trained local agents and supervisors for security evaluation.

    Returns:
        agents, supervisor_a, supervisor_b
    """
    agents: Dict[str, DDQNAgent] = {}

    for tls in env.tls_ids:
        agent = DDQNAgent(
            state_dim=env.get_state_dim(),
            action_dim=env.get_action_dim(),
            hidden_dim=128,
            learning_rate=1e-4,
            epsilon_start=0.0,
            epsilon_decay=1.0,
            epsilon_min=0.0,
        )
        ckpt_path = os.path.join(CHECKPOINT_DIR_AGENTS, f"{tls}_final.pth")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing agent checkpoint: {ckpt_path}")
        agent.load(ckpt_path)
        agent.epsilon = 0.0
        agents[tls] = agent

    supervisor_a = SupervisorAgent(
        group_tls_ids=env.group_a,
        state_dim_per_agent=env.get_local_state_dim(),
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.95,
        buffer_capacity=10_000,
        batch_size=64,
    )
    supervisor_b = SupervisorAgent(
        group_tls_ids=env.group_b,
        state_dim_per_agent=env.get_local_state_dim(),
        hidden_dim=64,
        learning_rate=0.001,
        gamma=0.95,
        buffer_capacity=10_000,
        batch_size=64,
    )

    sup_a_ckpt = os.path.join(CHECKPOINT_DIR_SUPERVISORS, "supervisor_a_final.pth")
    sup_b_ckpt = os.path.join(CHECKPOINT_DIR_SUPERVISORS, "supervisor_b_final.pth")
    if not os.path.exists(sup_a_ckpt):
        raise FileNotFoundError(f"Missing supervisor checkpoint: {sup_a_ckpt}")
    if not os.path.exists(sup_b_ckpt):
        raise FileNotFoundError(f"Missing supervisor checkpoint: {sup_b_ckpt}")

    supervisor_a.load(sup_a_ckpt)
    supervisor_b.load(sup_b_ckpt)

    return agents, supervisor_a, supervisor_b


def run_single_scenario(
    env: SupervisorSumoEnvironment,
    agents: Dict[str, DDQNAgent],
    supervisor_a: SupervisorAgent,
    supervisor_b: SupervisorAgent,
    security: SecurityLayer,
    episodes: int,
) -> Dict[str, object]:
    """
    Run one security scenario and collect episode-level metrics.

    Security insertion points:
    - after env.reset() local states
    - after each env.step() next local states
    """
    per_intersection_rewards = {tls: [] for tls in env.tls_ids}
    network_rewards = []
    wait_times = []

    for _ in range(episodes):
        local_states = env.reset()
        local_states = security.process(local_states, step=0)

        signals_a = supervisor_a.get_signals({tls: local_states[tls] for tls in env.group_a})
        signals_b = supervisor_b.get_signals({tls: local_states[tls] for tls in env.group_b})
        enhanced = env.build_enhanced_states(local_states, signals_a, signals_b)

        done = False
        step = 0
        ep_reward = {tls: 0.0 for tls in env.tls_ids}

        while not done:
            actions = {
                tls: agents[tls].select_action(enhanced[tls], training=False)
                for tls in env.tls_ids
            }

            next_local, rewards, done, info = env.step(actions)
            step += 1
            next_local = security.process(next_local, step=step)

            next_sig_a = supervisor_a.get_signals({tls: next_local[tls] for tls in env.group_a})
            next_sig_b = supervisor_b.get_signals({tls: next_local[tls] for tls in env.group_b})
            enhanced = env.build_enhanced_states(next_local, next_sig_a, next_sig_b)
            local_states = next_local

            for tls in env.tls_ids:
                ep_reward[tls] += float(rewards[tls])

            if done:
                wait_times.append(float(info["avg_waiting_time"]))

        for tls in env.tls_ids:
            per_intersection_rewards[tls].append(ep_reward[tls])

        network_rewards.append(float(sum(ep_reward.values())))

    metrics = security.get_metrics()

    return {
        "network_rewards": network_rewards,
        "per_intersection_rewards": per_intersection_rewards,
        "wait_times": wait_times,
        "security_metrics": metrics,
        "avg_network_reward": float(np.mean(network_rewards)) if network_rewards else 0.0,
        "std_network_reward": float(np.std(network_rewards)) if network_rewards else 0.0,
        "avg_wait_time": float(np.mean(wait_times)) if wait_times else 0.0,
    }
