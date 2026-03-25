from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Sequence

import numpy as np


@dataclass
class QoEWeights:
    rebuffer_penalty: float = 4.3
    smoothness_penalty: float = 1.0

    @classmethod
    def default(cls) -> "QoEWeights":
        return cls()


@dataclass
class EpisodeMetrics:
    average_bitrate_mbps: float
    average_quality_mbps: float
    total_rebuffer_s: float
    stall_ratio: float
    switch_count: int
    switch_magnitude_mbps: float
    mean_observed_throughput_mbps: float
    total_qoe: float
    steps: int

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self)



def step_reward(
    bitrate_kbps: int,
    previous_bitrate_kbps: int | None,
    rebuffer_s: float,
    weights: QoEWeights,
) -> tuple[float, float, float]:
    quality = bitrate_kbps / 1000.0
    smoothness = 0.0 if previous_bitrate_kbps is None else abs(bitrate_kbps - previous_bitrate_kbps) / 1000.0
    reward = quality - weights.rebuffer_penalty * rebuffer_s - weights.smoothness_penalty * smoothness
    return float(reward), float(quality), float(smoothness)



def summarize_episode(
    chosen_bitrates_kbps: Sequence[int],
    rebuffer_events_s: Sequence[float],
    rewards: Sequence[float],
    observed_throughputs_mbps: Sequence[float] | None = None,
    segment_duration_s: float = 4.0,
) -> EpisodeMetrics:
    bitrate_mbps = np.array(chosen_bitrates_kbps, dtype=np.float32) / 1000.0
    switches = np.abs(np.diff(bitrate_mbps)) if len(bitrate_mbps) > 1 else np.array([], dtype=np.float32)
    total_rebuffer = float(np.sum(rebuffer_events_s))
    total_qoe = float(np.sum(rewards))
    average_bitrate = float(bitrate_mbps.mean()) if len(bitrate_mbps) else 0.0
    total_play_time = float(len(bitrate_mbps) * segment_duration_s + total_rebuffer)
    throughput_mean = float(np.mean(observed_throughputs_mbps)) if observed_throughputs_mbps else 0.0
    return EpisodeMetrics(
        average_bitrate_mbps=average_bitrate,
        average_quality_mbps=average_bitrate,
        total_rebuffer_s=total_rebuffer,
        stall_ratio=float(total_rebuffer / max(total_play_time, 1e-6)),
        switch_count=int(np.count_nonzero(switches > 1e-6)),
        switch_magnitude_mbps=float(switches.sum()) if switches.size else 0.0,
        mean_observed_throughput_mbps=throughput_mean,
        total_qoe=total_qoe,
        steps=len(chosen_bitrates_kbps),
    )
