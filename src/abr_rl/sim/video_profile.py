from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import List

import numpy as np


@dataclass
class VideoProfile:
    segment_duration_s: float
    bitrate_kbps: List[int]
    total_segments: int
    initial_buffer_s: float
    max_buffer_s: float
    network_overhead: float = 1.05
    history_length: int = 8
    request_rtt_ms: float = 80.0
    content_profile: str = "mixed"
    content_variability: float = 0.18
    content_seed: int = 11

    @classmethod
    def default(cls) -> "VideoProfile":
        return cls(
            segment_duration_s=4.0,
            bitrate_kbps=[300, 750, 1200, 1850, 2850, 4300],
            total_segments=36,
            initial_buffer_s=8.0,
            max_buffer_s=36.0,
        )

    @property
    def bitrate_mbps(self) -> np.ndarray:
        return np.array(self.bitrate_kbps, dtype=np.float32) / 1000.0

    @property
    def chunk_sizes_bytes(self) -> np.ndarray:
        bitrate_bps = np.array(self.bitrate_kbps, dtype=np.float32) * 1000.0
        sizes = bitrate_bps * self.segment_duration_s / 8.0
        sizes *= self.network_overhead
        return sizes.astype(np.float32)

    @cached_property
    def segment_complexity(self) -> np.ndarray:
        rng = np.random.default_rng(self.content_seed)
        x = np.linspace(0, 2 * np.pi, self.total_segments)
        if self.content_profile == "sports":
            base = 1.08 + 0.15 * np.sin(2.5 * x) + 0.06 * np.cos(5.0 * x)
        elif self.content_profile == "documentary":
            base = 0.94 + 0.08 * np.sin(1.3 * x)
        else:
            base = 1.0 + 0.12 * np.sin(1.7 * x) + 0.05 * np.cos(3.2 * x)
        noise = rng.normal(0.0, self.content_variability, self.total_segments)
        multipliers = np.clip(base + noise, 0.72, 1.38)
        return multipliers.astype(np.float32)

    @cached_property
    def chunk_size_matrix_bytes(self) -> np.ndarray:
        codec_bias = np.linspace(0.98, 1.08, len(self.bitrate_kbps), dtype=np.float32)
        matrix = self.chunk_sizes_bytes[None, :] * self.segment_complexity[:, None] * codec_bias[None, :]
        return matrix.astype(np.float32)

    def chunk_sizes_for_segment(self, segment_index: int) -> np.ndarray:
        idx = int(np.clip(segment_index, 0, self.total_segments - 1))
        return self.chunk_size_matrix_bytes[idx]

    @property
    def average_chunk_sizes_bytes(self) -> np.ndarray:
        return self.chunk_size_matrix_bytes.mean(axis=0).astype(np.float32)
