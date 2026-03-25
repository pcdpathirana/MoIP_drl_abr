from __future__ import annotations

from dataclasses import dataclass, fields, asdict
from pathlib import Path
from typing import Any, List

import yaml


@dataclass
class VideoConfig:
    segment_duration_s: float
    total_segments: int
    initial_buffer_s: float
    max_buffer_s: float
    network_overhead: float
    bitrate_kbps: List[int]
    history_length: int = 8
    request_rtt_ms: float = 80.0
    content_profile: str = "mixed"
    content_variability: float = 0.18
    content_seed: int = 11


@dataclass
class RewardConfig:
    rebuffer_penalty: float = 4.3
    smoothness_penalty: float = 1.0


@dataclass
class TrainingConfig:
    total_timesteps: int = 20000
    timesteps_per_batch: int = 1024
    update_epochs: int = 8
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    learning_rate: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    checkpoint_every_updates: int = 5
    trace_sampling: str = "random"


@dataclass
class DataConfig:
    train_dir: str = "data/traces/research/train"
    val_dir: str = "data/traces/research/val"
    eval_dir: str = "data/traces/research/test"
    recursive: bool = True


@dataclass
class OutputConfig:
    run_dir: str = "results/professional_demo"


@dataclass
class ExperimentConfig:
    experiment_name: str
    seed: int
    video: VideoConfig
    reward: RewardConfig
    training: TrainingConfig
    data: DataConfig
    output: OutputConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)



def _construct_dataclass(cls, raw: dict[str, Any] | None):
    raw = raw or {}
    allowed = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in allowed}
    return cls(**filtered)



def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return ExperimentConfig(
        experiment_name=str(raw.get("experiment_name", path.stem)),
        seed=int(raw.get("seed", 42)),
        video=_construct_dataclass(VideoConfig, raw.get("video")),
        reward=_construct_dataclass(RewardConfig, raw.get("reward")),
        training=_construct_dataclass(TrainingConfig, raw.get("training")),
        data=_construct_dataclass(DataConfig, raw.get("data")),
        output=_construct_dataclass(OutputConfig, raw.get("output")),
    )
