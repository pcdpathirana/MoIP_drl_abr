from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from abr_rl.metrics.qoe import QoEWeights, summarize_episode, step_reward
from abr_rl.sim.video_profile import VideoProfile


@dataclass
class DownloadRecord:
    segment_index: int
    bitrate_kbps: int
    chunk_size_mb: float
    segment_complexity: float
    download_time_s: float
    observed_throughput_mbps: float
    rebuffer_s: float
    buffer_after_s: float
    reward: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass
class SimulationState:
    segment_index: int = 0
    clock_s: float = 0.0
    buffer_s: float = 0.0
    done: bool = False
    previous_bitrate_kbps: int | None = None
    throughput_history_mbps: List[float] = field(default_factory=list)
    download_history_s: List[float] = field(default_factory=list)
    records: List[DownloadRecord] = field(default_factory=list)

    def reset_histories(self, history_length: int) -> None:
        self.throughput_history_mbps = [0.0] * history_length
        self.download_history_s = [0.0] * history_length


class ABRSimulator:
    def __init__(self, trace_mbps: np.ndarray, profile: VideoProfile, reward_weights: QoEWeights):
        self.trace_mbps = np.asarray(trace_mbps, dtype=np.float32)
        self.profile = profile
        self.reward_weights = reward_weights
        self.state = SimulationState()
        self.reset()

    def set_trace(self, trace_mbps: np.ndarray) -> None:
        self.trace_mbps = np.asarray(trace_mbps, dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.state = SimulationState(
            segment_index=0,
            clock_s=0.0,
            buffer_s=float(self.profile.initial_buffer_s),
            done=False,
            previous_bitrate_kbps=None,
        )
        self.state.reset_histories(self.profile.history_length)
        return self.get_observation()

    def _throughput_at(self, time_s: float) -> float:
        idx = int(time_s) % len(self.trace_mbps)
        return max(float(self.trace_mbps[idx]), 0.05)

    def _simulate_download(self, chunk_size_bytes: float) -> tuple[float, float]:
        remaining_bits = float(chunk_size_bytes) * 8.0
        elapsed = self.profile.request_rtt_ms / 1000.0

        while remaining_bits > 1e-6:
            current_second = int(self.state.clock_s)
            offset = self.state.clock_s - current_second
            remaining_in_slot = max(1e-6, 1.0 - offset)
            throughput_bps = self._throughput_at(self.state.clock_s) * 1_000_000.0
            transferable_bits = throughput_bps * remaining_in_slot

            if transferable_bits >= remaining_bits:
                dt = remaining_bits / max(throughput_bps, 1.0)
                self.state.clock_s += dt
                elapsed += dt
                remaining_bits = 0.0
            else:
                self.state.clock_s += remaining_in_slot
                elapsed += remaining_in_slot
                remaining_bits -= transferable_bits

        observed_mbps = (chunk_size_bytes * 8.0) / max(elapsed, 1e-6) / 1_000_000.0
        return elapsed, observed_mbps

    def _apply_playback_during_download(self, download_time_s: float) -> float:
        if self.state.buffer_s >= download_time_s:
            self.state.buffer_s -= download_time_s
            return 0.0

        rebuffer = download_time_s - self.state.buffer_s
        self.state.buffer_s = 0.0
        return rebuffer

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool, Dict]:
        if self.state.done:
            raise RuntimeError("Episode already finished. Call reset().")

        action_index = int(np.clip(action_index, 0, len(self.profile.bitrate_kbps) - 1))
        bitrate_kbps = int(self.profile.bitrate_kbps[action_index])
        chunk_sizes = self.profile.chunk_sizes_for_segment(self.state.segment_index)
        chunk_size_bytes = float(chunk_sizes[action_index])
        segment_complexity = float(self.profile.segment_complexity[self.state.segment_index])

        download_time_s, observed_mbps = self._simulate_download(chunk_size_bytes)
        rebuffer_s = self._apply_playback_during_download(download_time_s)
        self.state.buffer_s = min(self.profile.max_buffer_s, self.state.buffer_s + self.profile.segment_duration_s)

        reward, quality_mbps, smoothness_mbps = step_reward(
            bitrate_kbps=bitrate_kbps,
            previous_bitrate_kbps=self.state.previous_bitrate_kbps,
            rebuffer_s=rebuffer_s,
            weights=self.reward_weights,
        )

        record = DownloadRecord(
            segment_index=self.state.segment_index,
            bitrate_kbps=bitrate_kbps,
            chunk_size_mb=float(chunk_size_bytes / 1_000_000.0),
            segment_complexity=segment_complexity,
            download_time_s=float(download_time_s),
            observed_throughput_mbps=float(observed_mbps),
            rebuffer_s=float(rebuffer_s),
            buffer_after_s=float(self.state.buffer_s),
            reward=float(reward),
        )
        self.state.records.append(record)

        self.state.throughput_history_mbps = self.state.throughput_history_mbps[1:] + [float(observed_mbps)]
        self.state.download_history_s = self.state.download_history_s[1:] + [float(download_time_s)]

        self.state.previous_bitrate_kbps = bitrate_kbps
        self.state.segment_index += 1
        self.state.done = self.state.segment_index >= self.profile.total_segments

        info = {
            "segment_index": int(record.segment_index),
            "bitrate_kbps": bitrate_kbps,
            "chunk_size_mb": float(record.chunk_size_mb),
            "segment_complexity": segment_complexity,
            "download_time_s": float(download_time_s),
            "observed_throughput_mbps": float(observed_mbps),
            "rebuffer_s": float(rebuffer_s),
            "buffer_s": float(self.state.buffer_s),
            "quality_mbps": float(quality_mbps),
            "smoothness_mbps": float(smoothness_mbps),
        }

        if self.state.done:
            chosen = [r.bitrate_kbps for r in self.state.records]
            rebuffers = [r.rebuffer_s for r in self.state.records]
            rewards = [r.reward for r in self.state.records]
            throughputs = [r.observed_throughput_mbps for r in self.state.records]
            info["episode_metrics"] = summarize_episode(
                chosen_bitrates_kbps=chosen,
                rebuffer_events_s=rebuffers,
                rewards=rewards,
                observed_throughputs_mbps=throughputs,
                segment_duration_s=self.profile.segment_duration_s,
            ).to_dict()

        return self.get_observation(), float(reward), self.state.done, info

    def get_observation(self) -> np.ndarray:
        next_chunk_sizes_mb = self.profile.chunk_sizes_for_segment(self.state.segment_index if not self.state.done else self.profile.total_segments - 1) / 1_000_000.0
        last_bitrate = 0.0 if self.state.previous_bitrate_kbps is None else self.state.previous_bitrate_kbps / max(self.profile.bitrate_kbps)
        remaining = (self.profile.total_segments - self.state.segment_index) / max(self.profile.total_segments, 1)

        obs = np.concatenate(
            [
                np.array(self.state.throughput_history_mbps, dtype=np.float32) / 10.0,
                np.array(self.state.download_history_s, dtype=np.float32) / max(self.profile.segment_duration_s * 2.0, 1.0),
                next_chunk_sizes_mb.astype(np.float32),
                np.array([
                    self.state.buffer_s / max(self.profile.max_buffer_s, 1.0),
                    last_bitrate,
                    remaining,
                ], dtype=np.float32),
            ]
        )
        return obs.astype(np.float32)

    def episode_dataframe(self) -> pd.DataFrame:
        rows = [record.to_dict() for record in self.state.records]
        return pd.DataFrame(rows)

    @property
    def observation_dim(self) -> int:
        return int(self.get_observation().shape[0])

    @property
    def action_dim(self) -> int:
        return len(self.profile.bitrate_kbps)
