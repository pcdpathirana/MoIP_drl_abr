from __future__ import annotations

import numpy as np

try:
    from gymnasium import spaces as gym_spaces  # type: ignore
    Box = gym_spaces.Box
    Discrete = gym_spaces.Discrete
except Exception:
    class Box:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype

        def sample(self):
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)

    class Discrete:
        def __init__(self, n: int):
            self.n = n

        def sample(self):
            return int(np.random.randint(0, self.n))
