"""
Security layer for traffic-state protection in security-phase experiments.

Step 5.1 scope:
- Define SecurityLayer class and initialization contract
- Accept all tunable parameters for attack/defense/network reliability
- Initialize rolling history, last-known state, and delay buffers
- Load pre-trained LSTM predictor checkpoint

Note:
- Full behavior routing and algorithmic processing is implemented in Step 5.2+
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

import numpy as np

from lstm_predictor import TrafficLSTM


SUPPORTED_MODES = {"baseline", "attack", "defense", "unreliable", "secure"}
DEFENSE_MODES = {"defense", "secure"}


@dataclass
class SecurityLayerConfig:
    """Container for security layer tunable parameters."""

    mode: str = "baseline"
    fdi_prob: float = 0.15
    fdi_min: float = 10.0
    fdi_max: float = 15.0
    window_size: int = 20
    z_threshold: float = 3.0
    packet_loss_prob: float = 0.05
    max_delay_steps: int = 3
    lstm_checkpoint: str = "checkpoints_security/lstm_predictor.pth"
    seed: int = 42


class SecurityLayer:
    """
    Security wrapper for raw local traffic states.

    Input format:
      {tls_id: np.ndarray shape (6,)}

    Output format:
      {tls_id: np.ndarray shape (6,)}

    Design constraints:
    - Queue features are indices [0..3]
    - Phase/timer features are indices [4..5] and must remain untouched by attack/defense
    """

    def __init__(
        self,
        mode: str = "baseline",
        fdi_prob: float = 0.15,
        fdi_min: float = 10.0,
        fdi_max: float = 15.0,
        window_size: int = 20,
        z_threshold: float = 3.0,
        packet_loss_prob: float = 0.05,
        max_delay_steps: int = 3,
        lstm_checkpoint: str = "checkpoints_security/lstm_predictor.pth",
        seed: int = 42,
    ) -> None:
        self.config = SecurityLayerConfig(
            mode=mode,
            fdi_prob=fdi_prob,
            fdi_min=fdi_min,
            fdi_max=fdi_max,
            window_size=window_size,
            z_threshold=z_threshold,
            packet_loss_prob=packet_loss_prob,
            max_delay_steps=max_delay_steps,
            lstm_checkpoint=lstm_checkpoint,
            seed=seed,
        )

        self._validate_config()

        self.rng = np.random.default_rng(seed=self.config.seed)

        # Initialized on first process(...) call based on observed tls_ids.
        self._tls_ids: Optional[List[str]] = None

        # Rolling queue history: tls_id -> deque of queue vectors shape (4,)
        self.history_buffers: Dict[str, Deque[np.ndarray]] = {}

        # Last successfully delivered raw state: tls_id -> np.ndarray shape (6,)
        self.last_known_states: Dict[str, np.ndarray] = {}

        # Delay buffers for unreliable mode:
        # tls_id -> deque of tuples (deliver_step, state_vector)
        self.delay_buffers: Dict[str, Deque[tuple[int, np.ndarray]]] = {}

        # Event logs and counters
        self.attack_log: List[dict] = []
        self.detection_log: List[dict] = []
        self.false_positive_count: int = 0

        self.total_attack_events: int = 0
        self.total_detection_events: int = 0

        self.lstm_model: Optional[TrafficLSTM] = self._load_lstm_model()

    def _validate_config(self) -> None:
        """Validate mode and numeric parameter ranges."""
        c = self.config

        if c.mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode '{c.mode}'. Allowed: {sorted(SUPPORTED_MODES)}")

        if not (0.0 <= c.fdi_prob <= 1.0):
            raise ValueError("fdi_prob must be in [0, 1]")

        if c.fdi_min < 0 or c.fdi_max < 0 or c.fdi_min > c.fdi_max:
            raise ValueError("FDI range must satisfy 0 <= fdi_min <= fdi_max")

        if c.window_size < 2:
            raise ValueError("window_size must be >= 2")

        if c.z_threshold <= 0:
            raise ValueError("z_threshold must be > 0")

        if not (0.0 <= c.packet_loss_prob <= 1.0):
            raise ValueError("packet_loss_prob must be in [0, 1]")

        if c.max_delay_steps < 0:
            raise ValueError("max_delay_steps must be >= 0")

    def _load_lstm_model(self) -> Optional[TrafficLSTM]:
        """Load LSTM checkpoint according to mode requirements."""
        ckpt = self.config.lstm_checkpoint

        if os.path.exists(ckpt):
            return TrafficLSTM.load(ckpt)

        if self.config.mode in DEFENSE_MODES:
            raise FileNotFoundError(
                f"LSTM checkpoint required for mode '{self.config.mode}' but not found: {ckpt}"
            )

        return None

    def _initialize_state_buffers(self, raw_states: Dict[str, np.ndarray]) -> None:
        """Initialize per-intersection buffers on first observation."""
        self._tls_ids = list(raw_states.keys())

        for tls in self._tls_ids:
            state = np.array(raw_states[tls], dtype=np.float32).copy()
            if state.shape != (6,):
                raise ValueError(f"State for {tls} must have shape (6,), got {state.shape}")

            self.history_buffers[tls] = deque(maxlen=self.config.window_size)
            self.last_known_states[tls] = state.copy()
            self.delay_buffers[tls] = deque()

    def _validate_input_states(self, raw_states: Dict[str, np.ndarray]) -> None:
        """Validate incoming raw state dict format."""
        if not isinstance(raw_states, dict) or not raw_states:
            raise ValueError("raw_states must be a non-empty dict: {tls_id: np.ndarray(6,)}")

        for tls, state in raw_states.items():
            arr = np.array(state)
            if arr.shape != (6,):
                raise ValueError(f"Invalid state shape for {tls}: expected (6,), got {arr.shape}")

    def _copy_states(self, raw_states: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Return deep-copied state dict to preserve caller immutability."""
        return {tls: np.array(state, dtype=np.float32).copy() for tls, state in raw_states.items()}

    def process(self, raw_states: Dict[str, np.ndarray], step: int) -> Dict[str, np.ndarray]:
        """
        Entry point for state processing.

        Step 5.1 behavior:
        - validates and initializes internal buffers
        - returns pass-through copy (algorithm routing added in Step 5.2)
        """
        _ = step  # used in later steps

        self._validate_input_states(raw_states)

        if self._tls_ids is None:
            self._initialize_state_buffers(raw_states)

        return self._copy_states(raw_states)
