from __future__ import annotations

import numpy as np


class BOLAInspiredController:
    """Readable BOLA-style baseline for research demos.

    This is intentionally lightweight. It captures the core idea:
    choose bitrate as a function of utility and current buffer level.
    """

    def __init__(self, bitrate_kbps: list[int], chunk_sizes_bytes: np.ndarray, max_buffer_s: float):
        self.bitrate_kbps = bitrate_kbps
        self.chunk_sizes_bytes = chunk_sizes_bytes.astype(np.float32)
        utilities = np.log(np.array(bitrate_kbps, dtype=np.float32) / min(bitrate_kbps))
        self.utilities = utilities
        self.gp = 5.0
        self.V = max_buffer_s / max(float(utilities[-1] + self.gp), 1e-6)

    def select_action(self, buffer_s: float, **kwargs) -> int:
        scores = (self.V * (self.utilities + self.gp) - buffer_s) / np.maximum(self.chunk_sizes_bytes / 1_000_000.0, 1e-6)
        idx = int(np.argmax(scores))
        if np.max(scores) < 0:
            return 0
        return idx
