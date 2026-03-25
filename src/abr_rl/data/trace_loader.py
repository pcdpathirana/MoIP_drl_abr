from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple
import json
import random

import numpy as np
import pandas as pd


SUPPORTED_EXTENSIONS = {".csv", ".txt", ".log", ".json"}


@dataclass
class TraceSummary:
    trace_name: str
    scenario: str
    samples: int
    duration_s: float
    mean_mbps: float
    std_mbps: float
    min_mbps: float
    max_mbps: float
    p10_mbps: float
    median_mbps: float
    p90_mbps: float
    outage_ratio: float
    variation_ratio: float

    def to_dict(self) -> dict[str, float | str | int]:
        return asdict(self)


class TraceRepository:
    def __init__(self, traces: Dict[str, np.ndarray], sampling: str = "random", seed: int = 42):
        if not traces:
            raise ValueError("TraceRepository requires at least one trace.")
        self.traces = {name: np.asarray(values, dtype=np.float32) for name, values in traces.items()}
        self.names = sorted(self.traces.keys())
        self.sampling = sampling
        self.rng = random.Random(seed)
        self._idx = 0

    def __len__(self) -> int:
        return len(self.names)

    def sample(self) -> tuple[str, np.ndarray]:
        if self.sampling == "round_robin":
            name = self.names[self._idx % len(self.names)]
            self._idx += 1
            return name, self.traces[name]
        name = self.rng.choice(self.names)
        return name, self.traces[name]

    def items(self) -> Iterator[tuple[str, np.ndarray]]:
        for name in self.names:
            yield name, self.traces[name]

    def catalog(self) -> pd.DataFrame:
        return build_trace_catalog(self.traces)



def infer_scenario_from_name(trace_name: str) -> str:
    name = trace_name.lower()
    keywords = [
        "hotel", "wifi", "conference", "broadband", "mobile", "commute",
        "handoff", "congested", "stable", "bursty", "mixed", "fcc",
        "norway", "pensieve", "synthetic", "research",
    ]
    for keyword in keywords:
        if keyword in name:
            return keyword
    return "generic"



def summarize_trace(trace_name: str, values: np.ndarray) -> TraceSummary:
    values = np.asarray(values, dtype=np.float32).flatten()
    values = np.clip(values, 0.0, None)
    if values.size == 0:
        raise ValueError(f"Trace {trace_name} is empty.")
    mean = float(values.mean())
    std = float(values.std())
    return TraceSummary(
        trace_name=trace_name,
        scenario=infer_scenario_from_name(trace_name),
        samples=int(values.size),
        duration_s=float(values.size),
        mean_mbps=mean,
        std_mbps=std,
        min_mbps=float(values.min()),
        max_mbps=float(values.max()),
        p10_mbps=float(np.percentile(values, 10)),
        median_mbps=float(np.percentile(values, 50)),
        p90_mbps=float(np.percentile(values, 90)),
        outage_ratio=float(np.mean(values < 0.5)),
        variation_ratio=float(std / max(mean, 1e-6)),
    )



def build_trace_catalog(traces: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = [summarize_trace(name, values).to_dict() for name, values in sorted(traces.items())]
    if not rows:
        return pd.DataFrame(columns=list(TraceSummary.__annotations__.keys()))
    return pd.DataFrame(rows).sort_values(["scenario", "trace_name"]).reset_index(drop=True)



def save_trace_catalog(traces: Dict[str, np.ndarray], output_path: str | Path) -> pd.DataFrame:
    df = build_trace_catalog(traces)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df



def load_trace_file(path: str | Path) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
        lower_map = {str(col).lower(): col for col in df.columns}
        if "bandwidth_mbps" in lower_map:
            values = df[lower_map["bandwidth_mbps"]].astype(float).to_numpy()
        else:
            numeric_cols = list(df.select_dtypes(include=["number"]).columns)
            if not numeric_cols:
                raise ValueError(f"No numeric columns found in trace CSV: {path}")
            preferred = None
            for col in numeric_cols:
                col_lower = str(col).lower()
                if any(token in col_lower for token in ["bandwidth", "throughput", "mbps", "kbps"]):
                    preferred = col
            target_col = preferred or numeric_cols[-1]
            values = df[target_col].astype(float).to_numpy()
            if "kbps" in str(target_col).lower():
                values = values / 1000.0
    elif suffix in {".txt", ".log"}:
        raw = np.loadtxt(path, dtype=float)
        values = raw[:, -1] if raw.ndim > 1 else raw
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            values = np.array(json.load(f), dtype=float)
    else:
        raise ValueError(f"Unsupported trace format: {path}")

    values = np.asarray(values, dtype=np.float32).flatten()
    values = np.clip(values, 0.05, None)
    if values.size == 0:
        raise ValueError(f"Trace is empty: {path}")
    return values



def list_trace_files(directory: str | Path, recursive: bool = False) -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        p for p in iterator
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )



def load_trace_directory(directory: str | Path, recursive: bool = False) -> Dict[str, np.ndarray]:
    traces: Dict[str, np.ndarray] = {}
    for path in list_trace_files(directory, recursive=recursive):
        trace_name = path.relative_to(directory).with_suffix("").as_posix().replace("/", "__")
        traces[trace_name] = load_trace_file(path)
    return traces



def split_traces(trace_dict: Dict[str, np.ndarray], ratio: float = 0.8) -> tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    names = sorted(trace_dict.keys())
    cutoff = max(1, int(len(names) * ratio))
    train_names = names[:cutoff]
    test_names = names[cutoff:]
    return (
        {name: trace_dict[name] for name in train_names},
        {name: trace_dict[name] for name in test_names},
    )
