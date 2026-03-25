from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def harmonic_mean(values: list[float], default: float = 1.0) -> float:
    cleaned = [max(v, 1e-6) for v in values if v > 0]
    if not cleaned:
        return default
    return len(cleaned) / sum(1.0 / v for v in cleaned)


def highest_safe_bitrate_index(estimate_kbps: float, bitrate_kbps: list[int]) -> int:
    idx = 0
    for i, bitrate in enumerate(bitrate_kbps):
        if bitrate <= estimate_kbps:
            idx = i
    return idx
