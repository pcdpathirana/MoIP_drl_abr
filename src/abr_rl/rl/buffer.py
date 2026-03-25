from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)

    def add(self, obs, action, reward, done, log_prob, value) -> None:
        self.observations.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))

    def clear(self) -> None:
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.actions)

    def to_tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "obs": torch.tensor(np.array(self.observations, dtype=np.float32), dtype=torch.float32, device=device),
            "actions": torch.tensor(self.actions, dtype=torch.long, device=device),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32, device=device),
            "dones": torch.tensor(self.dones, dtype=torch.float32, device=device),
            "log_probs": torch.tensor(self.log_probs, dtype=torch.float32, device=device),
            "values": torch.tensor(self.values, dtype=torch.float32, device=device),
        }
