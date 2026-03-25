from __future__ import annotations

from abr_rl.baselines.common import harmonic_mean, highest_safe_bitrate_index


class FESTIVEInspiredController:
    """Simplified FESTIVE-style baseline.

    It uses harmonic throughput estimation, conservative up-switching,
    and sticky behavior to avoid unnecessary oscillations.
    """

    def __init__(self, bitrate_kbps: list[int], safety_factor: float = 0.85):
        self.bitrate_kbps = bitrate_kbps
        self.safety_factor = safety_factor
        self.last_action = 0

    def reset(self) -> None:
        self.last_action = 0

    def select_action(self, throughput_history_mbps: list[float], buffer_s: float, **kwargs) -> int:
        est_mbps = harmonic_mean(throughput_history_mbps[-5:], default=1.5)
        safe_kbps = est_mbps * 1000.0 * self.safety_factor
        candidate = highest_safe_bitrate_index(safe_kbps, self.bitrate_kbps)
        current = self.last_action

        if candidate > current:
            if buffer_s > 8.0 and safe_kbps > self.bitrate_kbps[current] * 1.2:
                current = min(current + 1, candidate)
        elif candidate < current:
            if buffer_s < 6.0 or safe_kbps < self.bitrate_kbps[current] * 0.9:
                current = candidate

        self.last_action = current
        return current
