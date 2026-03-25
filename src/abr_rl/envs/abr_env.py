from __future__ import annotations

from typing import Any, Dict
import numpy as np

try:
    import gymnasium as gym  # type: ignore
except Exception:
    class _Env:
        pass
    class _GymModule:
        Env = _Env
    gym = _GymModule()

from abr_rl.data.trace_loader import TraceRepository
from abr_rl.metrics.qoe import QoEWeights
from abr_rl.sim.abr_simulator import ABRSimulator
from abr_rl.sim.video_profile import VideoProfile
from abr_rl.spaces import Box, Discrete


class ABRStreamingEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(
        self,
        trace_mbps: np.ndarray | None = None,
        profile: VideoProfile | None = None,
        reward_weights: QoEWeights | None = None,
        trace_repository: TraceRepository | None = None,
    ):
        super().__init__()
        if profile is None or reward_weights is None:
            raise ValueError("profile and reward_weights are required")
        if trace_mbps is None and trace_repository is None:
            raise ValueError("Either trace_mbps or trace_repository must be provided")

        self.trace_repository = trace_repository
        self.current_trace_name = "single_trace"
        if trace_mbps is None and trace_repository is not None:
            self.current_trace_name, trace_mbps = trace_repository.sample()
        self.simulator = ABRSimulator(trace_mbps=np.asarray(trace_mbps, dtype=np.float32), profile=profile, reward_weights=reward_weights)
        self.observation_space = Box(low=0.0, high=10.0, shape=(self.simulator.observation_dim,), dtype=np.float32)
        self.action_space = Discrete(self.simulator.action_dim)

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None) -> tuple[np.ndarray, Dict[str, Any]]:
        del seed, options
        if self.trace_repository is not None:
            self.current_trace_name, trace = self.trace_repository.sample()
            self.simulator.set_trace(trace)
        obs = self.simulator.reset()
        return obs, {"trace_name": self.current_trace_name}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, done, info = self.simulator.step(int(action))
        truncated = False
        info["trace_name"] = self.current_trace_name
        return obs, reward, done, truncated, info
