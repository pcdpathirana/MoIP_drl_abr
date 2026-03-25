from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from abr_rl.config import load_config
from abr_rl.data.trace_loader import TraceRepository, load_trace_directory, save_trace_catalog
from abr_rl.envs.abr_env import ABRStreamingEnv
from abr_rl.metrics.qoe import QoEWeights
from abr_rl.rl.ppo import PPOTrainer
from abr_rl.sim.video_profile import VideoProfile
from abr_rl.utils.io import ensure_dir, save_json
from abr_rl.utils.plotting import plot_trace_examples, plot_training_history



def build_env_fn(config):
    traces = load_trace_directory(config.data.train_dir, recursive=config.data.recursive)
    if not traces:
        raise FileNotFoundError(f"No training traces found in {config.data.train_dir}")
    trace_repository = TraceRepository(
        traces,
        sampling=config.training.trace_sampling,
        seed=config.seed,
    )
    profile = VideoProfile(**config.video.__dict__)
    reward_weights = QoEWeights(**config.reward.__dict__)

    def env_fn():
        return ABRStreamingEnv(
            profile=profile,
            reward_weights=reward_weights,
            trace_repository=trace_repository,
        )

    return env_fn, traces



def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on the ABR environment.")
    parser.add_argument("--config", default="configs/professional_demo.yaml", help="Path to YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = ensure_dir(config.output.run_dir)
    save_json(Path(out_dir) / "run_config.json", config.to_dict())

    env_fn, train_traces = build_env_fn(config)
    catalog_df = save_trace_catalog(train_traces, Path(out_dir) / "train_trace_catalog.csv")
    plot_trace_examples(Path(out_dir) / "train_trace_catalog.csv", Path(out_dir) / "train_trace_scenarios.png")

    trainer = PPOTrainer(env_fn=env_fn, config=config)
    model_path, history_df = trainer.train()
    plot_training_history(Path(config.output.run_dir) / "training_history.csv", Path(config.output.run_dir) / "training_curve.png")

    if config.data.val_dir:
        val_traces = load_trace_directory(config.data.val_dir, recursive=config.data.recursive)
        if val_traces:
            save_trace_catalog(val_traces, Path(out_dir) / "val_trace_catalog.csv")

    save_json(
        Path(out_dir) / "training_summary.json",
        {
            "experiment_name": config.experiment_name,
            "training_traces": len(train_traces),
            "episodes_recorded": int(history_df["episodes_finished"].max()) if not history_df.empty else 0,
            "best_episode_reward_mean": float(history_df["episode_reward_mean"].max()) if not history_df.empty else 0.0,
            "best_train_qoe_mean": float(history_df["train_qoe_mean_last10"].max()) if "train_qoe_mean_last10" in history_df else 0.0,
        },
    )

    print(f"Training completed. Final model saved at: {model_path}")
    print(f"Training traces catalogued: {len(catalog_df)}")
    print(history_df.tail())


if __name__ == "__main__":
    main()
