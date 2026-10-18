"""Action-conditioned Double DQN supervisor with 3**4 joint advisory actions."""
from __future__ import annotations

from collections import deque
from itertools import product
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SIGNAL_ACTIONS = np.asarray(list(product((-1., 0., 1.), repeat=4)), dtype=np.float32)
SUPERVISOR_SCHEMA = 'joint-ddqn-81-v1'


class SupervisorNetwork(nn.Module):
    def __init__(self, input_dim=24, hidden_dim=64, output_dim=81):
        super().__init__()
        if output_dim != 81:
            raise ValueError('The corrected supervisor has 81 joint Q-actions')
        self.fc1, self.fc2, self.fc3 = nn.Linear(input_dim, hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim, output_dim)
        for layer in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))


class SupervisorReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        if not 0 <= int(action) < 81:
            raise ValueError('Joint action must be in [0, 80]')
        self.buffer.append((np.array(state, dtype=np.float32, copy=True), int(action), float(reward),
                            np.array(next_state, dtype=np.float32, copy=True), float(done)))

    def sample(self, size):
        s, a, r, ns, d = zip(*random.sample(self.buffer, size))
        return (torch.tensor(np.asarray(s)), torch.tensor(a).unsqueeze(1),
                torch.tensor(r, dtype=torch.float32).unsqueeze(1), torch.tensor(np.asarray(ns)),
                torch.tensor(d, dtype=torch.float32).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)


class SupervisorAgent:
    def __init__(self, group_tls_ids, state_dim_per_agent=6, hidden_dim=64,
                 learning_rate=5e-4, gamma=.95, buffer_capacity=10_000,
                 batch_size=64, global_summary_dim=0):
        self.group_tls_ids = list(group_tls_ids)
        if len(self.group_tls_ids) != 4 or global_summary_dim not in (0, 4):
            raise ValueError('Four local agents and zero/four summary features are required')
        self.local_input_dim = 4 * state_dim_per_agent
        self.global_summary_dim = global_summary_dim
        self.input_dim = self.local_input_dim + global_summary_dim
        self.output_dim, self.gamma, self.batch_size = 81, gamma, batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.online_net = SupervisorNetwork(self.input_dim, hidden_dim).to(self.device)
        self.target_net = SupervisorNetwork(self.input_dim, hidden_dim).to(self.device)
        self.update_target_network()
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=learning_rate)
        self.memory = SupervisorReplayBuffer(buffer_capacity)
        self.train_step, self.decision_step = 0, 0
        self.epsilon = 1.

    def _build_group_state(self, states, other_summary=None):
        state = np.concatenate([states[tls] for tls in self.group_tls_ids]).astype(np.float32)
        if self.global_summary_dim:
            if other_summary is None:
                raise ValueError('Communicating supervisor requires the other group summary')
            state = np.concatenate([state, np.asarray(other_summary, dtype=np.float32)])
        if state.shape != (self.input_dim,):
            raise ValueError(f'Expected supervisor input ({self.input_dim},), got {state.shape}')
        return state

    @staticmethod
    def encode_signals(signals):
        values = tuple(float(v) for v in signals)
        if len(values) != 4 or any(v not in (-1., 0., 1.) for v in values):
            raise ValueError('Four ternary advisory signals are required')
        return int(sum((int(v) + 1) * 3 ** (3 - i) for i, v in enumerate(values)))

    def select_action(self, local_states, other_summary=None, training=False):
        state = self._build_group_state(local_states, other_summary)
        if training:
            self.epsilon = 1. - .95 * min(1., self.decision_step / 50_000)
            self.decision_step += 1
        if training and random.random() < self.epsilon:
            return random.randrange(81)
        with torch.no_grad():
            return int(self.online_net(torch.as_tensor(state, device=self.device).unsqueeze(0)).argmax(1).item())

    def signals_for_action(self, action):
        return dict(zip(self.group_tls_ids, SIGNAL_ACTIONS[int(action)].tolist()))

    def get_signals(self, local_states_dict, training=False):
        return self.signals_for_action(self.select_action(local_states_dict, training=training))

    def get_signals_with_global(self, local_states_dict, other_summary, training=False):
        return self.signals_for_action(self.select_action(local_states_dict, other_summary, training))

    def store(self, local_states_dict, group_rewards, next_local_states_dict, done, *, action):
        reward = .01 * float(np.mean([group_rewards[tls] for tls in self.group_tls_ids]))
        self.memory.store(self._build_group_state(local_states_dict), action, reward,
                          self._build_group_state(next_local_states_dict), done)

    def store_with_global(self, local_states_dict, other_summary, group_rewards,
                          next_local_states_dict, next_other_summary, done, *, action):
        reward = .01 * float(np.mean([group_rewards[tls] for tls in self.group_tls_ids]))
        self.memory.store(self._build_group_state(local_states_dict, other_summary), action, reward,
                          self._build_group_state(next_local_states_dict, next_other_summary), done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = (v.to(self.device) for v in self.memory.sample(self.batch_size))
        values = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1, keepdim=True)
            targets = rewards + (1 - dones) * self.gamma * self.target_net(next_states).gather(1, next_actions)
        loss = F.smooth_l1_loss(values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.)
        self.optimizer.step()
        self.train_step += 1
        if self.train_step % 2000 == 0:
            self.update_target_network()
        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def state_dict(self):
        return {'schema': SUPERVISOR_SCHEMA, 'input_dim': self.input_dim,
                'online_net': self.online_net.state_dict(), 'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(), 'train_step': self.train_step,
                'decision_step': self.decision_step, 'epsilon': self.epsilon,
                'replay': list(self.memory.buffer), 'capacity': self.memory.buffer.maxlen}

    def load_state_dict(self, state, resume=True):
        if state.get('schema') != SUPERVISOR_SCHEMA or state.get('input_dim') != self.input_dim:
            raise ValueError('Incompatible supervisor: legacy tanh checkpoints cannot train a joint-action supervisor')
        self.online_net.load_state_dict(state['online_net'])
        self.target_net.load_state_dict(state['target_net'])
        if resume:
            self.optimizer.load_state_dict(state['optimizer'])
            self.train_step, self.decision_step, self.epsilon = state['train_step'], state['decision_step'], state['epsilon']
            self.memory.buffer = deque(state['replay'], maxlen=state['capacity'])

    def initialize_from_local(self, local):
        if self.input_dim != 28 or local.input_dim != 24:
            raise ValueError('Transfer is supported only from local 24-input to communicating 28-input')
        state = {k: v.clone() for k, v in local.online_net.state_dict().items()}
        weight = torch.zeros_like(self.online_net.fc1.weight)
        weight[:, :24] = state['fc1.weight']
        state['fc1.weight'] = weight
        self.online_net.load_state_dict(state)
        self.update_target_network()

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path, resume=False):
        # Corrected bundles contain NumPy replay arrays; only use trusted local files.
        self.load_state_dict(torch.load(path, map_location=self.device, weights_only=False), resume=resume)
