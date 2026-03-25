from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd


RESEARCH_SCENARIOS = [
    "broadband_stable",
    "broadband_bursty",
    "hotel_wifi",
    "conference_peak",
    "commute_mobile",
    "handoff_dropout",
    "home_wifi",
    "mixed_research",
]



def write_trace_csv(path: str | Path, values: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"bandwidth_mbps": np.asarray(values, dtype=float)})
    df.to_csv(path, index=False)



def step_trace(length: int, low: float, high: float, switch_at: int) -> np.ndarray:
    values = np.empty(length, dtype=np.float32)
    values[:switch_at] = low
    values[switch_at:] = high
    return values



def bursty_trace(length: int, base: float, noise: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = base + rng.normal(0.0, noise, length)
    for i in range(0, length, 12):
        if rng.random() < 0.5:
            values[i:i + 3] *= rng.uniform(0.25, 0.65)
    return np.clip(values, 0.2, None).astype(np.float32)



def mobile_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 5 * math.pi, length)
    seasonal = 4.0 + 2.2 * np.sin(t)
    noise = rng.normal(0.0, 0.7, length)
    values = seasonal + noise
    for i in range(8, length, 25):
        values[i:i + 4] *= rng.uniform(0.1, 0.35)
    return np.clip(values, 0.15, None).astype(np.float32)



def stable_trace(length: int, base: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = base + rng.normal(0.0, 0.25, length)
    return np.clip(values, 0.3, None).astype(np.float32)



def mixed_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    blocks = []
    remaining = length
    while remaining > 0:
        block_len = min(remaining, int(rng.integers(8, 20)))
        profile = rng.choice(["stable", "bursty", "drop"])
        if profile == "stable":
            block = stable_trace(block_len, float(rng.uniform(2.0, 9.0)), int(rng.integers(0, 100000)))
        elif profile == "bursty":
            block = bursty_trace(block_len, float(rng.uniform(2.5, 8.5)), float(rng.uniform(0.4, 1.5)), int(rng.integers(0, 100000)))
        else:
            block = stable_trace(block_len, float(rng.uniform(2.0, 7.0)), int(rng.integers(0, 100000)))
            start = int(rng.integers(0, max(1, block_len - 3)))
            block[start:start + 3] *= rng.uniform(0.15, 0.4)
        blocks.append(block)
        remaining -= block_len
    return np.concatenate(blocks)[:length].astype(np.float32)



def broadband_stable_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = float(rng.uniform(4.0, 7.0))
    trend = np.linspace(0.0, float(rng.uniform(-0.8, 0.8)), length)
    noise = rng.normal(0.0, 0.35, length)
    values = base + trend + noise
    return np.clip(values, 0.35, None).astype(np.float32)



def broadband_bursty_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = bursty_trace(length, float(rng.uniform(4.5, 6.5)), float(rng.uniform(0.7, 1.2)), seed)
    for i in range(10, length, 40):
        values[i:i + 5] *= rng.uniform(0.35, 0.7)
    return np.clip(values, 0.2, None).astype(np.float32)



def hotel_wifi_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 6 * math.pi, length)
    base = 4.2 + 1.1 * np.sin(t)
    noise = rng.normal(0.0, 0.55, length)
    values = base + noise
    for i in range(6, length, 18):
        values[i:i + 2] *= rng.uniform(0.2, 0.45)
    values[::45] += rng.uniform(0.8, 1.6)
    return np.clip(values, 0.15, None).astype(np.float32)



def conference_peak_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    midpoint = length // 2
    left = stable_trace(midpoint, float(rng.uniform(5.0, 6.5)), seed + 1)
    right = stable_trace(length - midpoint, float(rng.uniform(2.0, 3.5)), seed + 2)
    values = np.concatenate([left, right])
    for i in range(midpoint - 30, min(length, midpoint + 40), 10):
        values[i:i + 3] *= rng.uniform(0.25, 0.55)
    return np.clip(values, 0.15, None).astype(np.float32)



def commute_mobile_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = mobile_trace(length, seed)
    tunnel_starts = [int(length * 0.22), int(length * 0.57), int(length * 0.81)]
    for start in tunnel_starts:
        width = int(rng.integers(4, 9))
        values[start:start + width] *= rng.uniform(0.05, 0.2)
    return np.clip(values, 0.08, None).astype(np.float32)



def handoff_dropout_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = stable_trace(length, float(rng.uniform(4.5, 6.5)), seed)
    for start in [int(length * 0.18), int(length * 0.49), int(length * 0.76)]:
        width = int(rng.integers(2, 5))
        values[start:start + width] = rng.uniform(0.05, 0.25, width)
        if start + width < length:
            values[start + width:start + width + 6] = np.linspace(values[start + width - 1], rng.uniform(3.2, 5.8), min(6, length - (start + width)))
    values += rng.normal(0.0, 0.35, length)
    return np.clip(values, 0.05, None).astype(np.float32)



def home_wifi_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = stable_trace(length, float(rng.uniform(3.5, 5.5)), seed)
    contention = 0.8 + 0.2 * np.sin(np.linspace(0, 12 * math.pi, length))
    values = values * contention
    for i in range(15, length, 32):
        values[i:i + 4] *= rng.uniform(0.3, 0.7)
    return np.clip(values, 0.12, None).astype(np.float32)



def mixed_research_trace(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    segments = [
        broadband_stable_trace(length // 4, seed + 1),
        hotel_wifi_trace(length // 4, seed + 2),
        commute_mobile_trace(length // 4, seed + 3),
        broadband_bursty_trace(length - 3 * (length // 4), seed + 4),
    ]
    values = np.concatenate(segments)
    values += rng.normal(0.0, 0.2, len(values))
    return np.clip(values, 0.08, None).astype(np.float32)



def generate_research_trace(scenario: str, length: int, seed: int) -> np.ndarray:
    scenario = scenario.lower()
    generators = {
        "broadband_stable": broadband_stable_trace,
        "broadband_bursty": broadband_bursty_trace,
        "hotel_wifi": hotel_wifi_trace,
        "conference_peak": conference_peak_trace,
        "commute_mobile": commute_mobile_trace,
        "handoff_dropout": handoff_dropout_trace,
        "home_wifi": home_wifi_trace,
        "mixed_research": mixed_research_trace,
    }
    if scenario not in generators:
        raise ValueError(f"Unknown research scenario: {scenario}")
    return generators[scenario](length, seed)
