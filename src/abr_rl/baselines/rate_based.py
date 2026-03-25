from __future__ import annotations

from abr_rl.baselines.common import harmonic_mean, highest_safe_bitrate_index


class RateBasedController:
    def __init__(self, bitrate_kbps: list[int], safety_factor: float = 0.9):
        self.bitrate_kbps = bitrate_kbps
        self.safety_factor = safety_factor

    def select_action(self, throughput_history_mbps: list[float], **kwargs) -> int:
        est_mbps = harmonic_mean(throughput_history_mbps[-5:], default=1.5)
        safe_kbps = est_mbps * 1000.0 * self.safety_factor
        return highest_safe_bitrate_index(safe_kbps, self.bitrate_kbps)
