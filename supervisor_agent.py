"""
Supervisor Agent for Local Group Coordination
==============================================
Manages one group of 4 intersections.
- Observes all 4 agents' local states (4 × 6 = 24-dim input)
- Outputs 4 coordination signals in [-1, +1] (one per agent)
- Trains using group average reward via TD learning
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import os


# ─────────────────────────────────────────────────────────────────────
# 1. Neural Network
# ─────────────────────────────────────────────────────────────────────
class SupervisorNetwork(nn.Module):
    """
    3-layer MLP for the supervisor.
    Input : concatenated states of all agents in group (24-dim)
    Output: coordination signals via tanh → range [-1, +1]
    """
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=4):
        super(SupervisorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))   # Signals in [-1, +1]
        return x


# ─────────────────────────────────────────────────────────────────────
# 2. Replay Buffer
# ─────────────────────────────────────────────────────────────────────
class SupervisorReplayBuffer:
    """
    Experience replay for the supervisor.
    Stores group-level transitions: (24-dim state, reward, 24-dim next_state, done)
    """
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def store(self, group_state, group_reward, next_group_state, done):
        self.buffer.append((
            np.array(group_state,      dtype=np.float32),
            float(group_reward),
            np.array(next_group_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────────────────────────────────
# 3. Supervisor Agent
# ─────────────────────────────────────────────────────────────────────
class SupervisorAgent:
    """
    Supervisor for one group of 4 intersections.

    Responsibilities:
      - Collect all local states → build 24-dim group state
      - Output 4 coordination signals (one per agent)
      - Train on group average reward via TD learning
      - Update target network periodically

    Args:
        group_tls_ids      : list of 4 TLS IDs in this group, e.g. ['tls_1','tls_2','tls_3','tls_4']
        state_dim_per_agent: local state dimension per agent (default 6)
        hidden_dim         : hidden layer size (default 64)
        learning_rate      : Adam LR (default 0.001)
        gamma              : discount factor (default 0.95)
        buffer_capacity    : replay buffer size (default 10 000)
        batch_size         : training batch size (default 64)
    """

    def __init__(self,
                 group_tls_ids,
                 state_dim_per_agent=6,
                 hidden_dim=64,
                 learning_rate=0.001,
                 gamma=0.95,
                 buffer_capacity=10_000,
                 batch_size=64):

        self.group_tls_ids      = group_tls_ids
        self.group_size         = len(group_tls_ids)           # 4
        self.input_dim          = self.group_size * state_dim_per_agent  # 24
        self.output_dim         = self.group_size              # 4
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.train_step         = 0

        # ── Device ──────────────────────────────────────────────────
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f"  [Supervisor] GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  [Supervisor] Using CPU")

        # ── Networks ────────────────────────────────────────────────
        self.online_net = SupervisorNetwork(self.input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_net = SupervisorNetwork(self.input_dim, hidden_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # ── Optimizer ───────────────────────────────────────────────
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)

        # ── Replay buffer ───────────────────────────────────────────
        self.memory = SupervisorReplayBuffer(capacity=buffer_capacity)

    # ── Helpers ─────────────────────────────────────────────────────
    def _build_group_state(self, local_states_dict):
        """
        Concatenate each agent's 6-dim local state → 24-dim group state.
        Order follows self.group_tls_ids.
        """
        return np.concatenate([local_states_dict[tls] for tls in self.group_tls_ids],
                              dtype=np.float32)

    # ── Core API ────────────────────────────────────────────────────
    def get_signals(self, local_states_dict):
        """
        Forward pass: generate coordination signals for each agent.

        Args:
            local_states_dict: {tls_id: np.array shape (6,)}

        Returns:
            signals_dict: {tls_id: float in [-1, +1]}
              Negative → low urgency ("you are fine")
              Positive → high urgency ("you are the bottleneck")
        """
        group_state  = self._build_group_state(local_states_dict)
        state_tensor = torch.FloatTensor(group_state).unsqueeze(0).to(self.device)

        self.online_net.eval()
        with torch.no_grad():
            signals = self.online_net(state_tensor).squeeze(0).cpu().numpy()
        self.online_net.train()

        return {tls: float(signals[i]) for i, tls in enumerate(self.group_tls_ids)}

    def store(self, local_states_dict, group_reward, next_local_states_dict, done):
        """
        Store one group-level transition in replay buffer.

        Args:
            local_states_dict      : current  {tls_id: 6-dim state}
            group_reward           : scalar average reward of the group
            next_local_states_dict : next     {tls_id: 6-dim state}
            done                   : episode end flag
        """
        group_state      = self._build_group_state(local_states_dict)
        next_group_state = self._build_group_state(next_local_states_dict)
        self.memory.store(group_state, group_reward, next_group_state, done)

    def train(self):
        """
        One gradient update for the supervisor.

        Training signal:
          current_signals = online_net(state)           # (batch, 4)
          td_target       = r + γ × mean(target_net(s')) broadcast to (batch, 4)
          loss            = MSE(current_signals, td_target)

        Returns: loss value (float) or None if buffer too small.
        """
        if len(self.memory) < self.batch_size:
            return None

        states, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states      = states.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # Current predicted signals
        current_signals = self.online_net(states)                            # (B, 4)

        # TD target
        with torch.no_grad():
            next_signals = self.target_net(next_states)                      # (B, 4)
            # Scalar target per sample → broadcast across 4 signal outputs
            td_target = rewards + (1 - dones) * self.gamma * next_signals.mean(dim=1, keepdim=True)
            td_target = td_target.expand_as(current_signals)                 # (B, 4)

        loss = F.mse_loss(current_signals, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_step += 1
        return loss.item()

    def update_target_network(self):
        """Hard copy: online → target."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Persistence ─────────────────────────────────────────────────
    def save(self, path):
        """Save supervisor checkpoint."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'train_step': self.train_step,
        }, path)
        print(f"  ✓ Supervisor saved → {path}")

    def load(self, path):
        """Load supervisor checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(ckpt['online_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.train_step = ckpt.get('train_step', 0)
        print(f"  ✓ Supervisor loaded ← {path}")


# ─────────────────────────────────────────────────────────────────────
# Quick smoke-test (run this file directly to verify)
# ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== SupervisorAgent smoke test ===\n")

    group_ids = ['tls_1', 'tls_2', 'tls_3', 'tls_4']
    sup = SupervisorAgent(group_tls_ids=group_ids)

    # Fake local states (6-dim each)
    fake_states = {tls: np.random.rand(6).astype(np.float32) for tls in group_ids}

    # Test get_signals
    signals = sup.get_signals(fake_states)
    print("Coordination signals:")
    for tls, sig in signals.items():
        print(f"  {tls}: {sig:+.4f}")

    # Test store + train (need ≥ 64 samples)
    fake_next = {tls: np.random.rand(6).astype(np.float32) for tls in group_ids}
    for _ in range(100):
        sup.store(fake_states, group_reward=-250.0, next_local_states_dict=fake_next, done=False)

    loss = sup.train()
    print(f"\nTraining loss: {loss:.6f}")

    # Test save / load
    sup.save('checkpoints_supervisor/test_supervisor.pth')
    sup.load('checkpoints_supervisor/test_supervisor.pth')

    print("\n✅ All checks passed!")
