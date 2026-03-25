from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import pandas as pd
import torch

from abr_rl.baselines.bola import BOLAInspiredController
from abr_rl.baselines.festive import FESTIVEInspiredController
from abr_rl.baselines.rate_based import RateBasedController
from abr_rl.config import load_config
from abr_rl.data.trace_loader import load_trace_directory, save_trace_catalog
from abr_rl.metrics.qoe import QoEWeights
from abr_rl.rl.ppo import load_model, ppo_action
from abr_rl.sim.abr_simulator import ABRSimulator
from abr_rl.sim.video_profile import VideoProfile
from abr_rl.utils.plotting import plot_algorithm_comparison



def run_controller(trace_name, trace, profile, reward_weights, controller_name, controller, session_root: Path):
    simulator = ABRSimulator(trace_mbps=trace, profile=profile, reward_weights=reward_weights)
    simulator.reset()

    while True:
        if controller_name == "bola":
            action = controller.select_action(buffer_s=simulator.state.buffer_s)
        elif controller_name == "festive":
            action = controller.select_action(
                throughput_history_mbps=simulator.state.throughput_history_mbps,
                buffer_s=simulator.state.buffer_s,
            )
        elif controller_name == "rate_based":
            action = controller.select_action(
                throughput_history_mbps=simulator.state.throughput_history_mbps,
            )
        else:
            raise ValueError(controller_name)

        obs, reward, done, info = simulator.step(action)
        if done:
            session_df = simulator.episode_dataframe()
            session_df.insert(0, "trace", trace_name)
            session_df.insert(1, "algorithm", controller_name)
            session_path = session_root / controller_name / f"{trace_name}.csv"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_df.to_csv(session_path, index=False)
            return info["episode_metrics"]



def run_ppo(trace_name, trace, profile, reward_weights, model, session_root: Path):
    simulator = ABRSimulator(trace_mbps=trace, profile=profile, reward_weights=reward_weights)
    obs = simulator.reset()
    device = next(model.parameters()).device
    while True:
        action = ppo_action(model, obs, device=device)
        obs, reward, done, info = simulator.step(action)
        if done:
            session_df = simulator.episode_dataframe()
            session_df.insert(0, "trace", trace_name)
            session_df.insert(1, "algorithm", "ppo")
            session_path = session_root / "ppo" / f"{trace_name}.csv"
            session_path.parent.mkdir(parents=True, exist_ok=True)
            session_df.to_csv(session_path, index=False)
            return info["episode_metrics"]



def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate baselines and PPO.")
    parser.add_argument("--config", default="configs/professional_demo.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    profile = VideoProfile(**config.video.__dict__)
    reward_weights = QoEWeights(**config.reward.__dict__)
    traces = load_trace_directory(config.data.eval_dir, recursive=config.data.recursive)
    if not traces:
        raise FileNotFoundError(f"No evaluation traces found in {config.data.eval_dir}")

    results = []
    out_dir = Path(config.output.run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    session_root = out_dir / "sessions"

    save_trace_catalog(traces, out_dir / "eval_trace_catalog.csv")

    bola = BOLAInspiredController(
        bitrate_kbps=profile.bitrate_kbps,
        chunk_sizes_bytes=profile.average_chunk_sizes_bytes,
        max_buffer_s=profile.max_buffer_s,
    )
    festive = FESTIVEInspiredController(profile.bitrate_kbps)
    rate_based = RateBasedController(profile.bitrate_kbps)

    model_path = Path(config.output.run_dir) / "ppo_final.pt"
    ppo_model = None
    if model_path.exists():
        obs_dim = ABRSimulator(next(iter(traces.values())), profile, reward_weights).observation_dim
        action_dim = len(profile.bitrate_kbps)
        ppo_model = load_model(
            model_path=model_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=config.training.hidden_dim,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    for trace_name, trace in traces.items():
        festive.reset()
        metrics = run_controller(trace_name, trace, profile, reward_weights, "bola", bola, session_root)
        results.append({"trace": trace_name, "algorithm": "BOLA-inspired", **metrics})

        festive.reset()
        metrics = run_controller(trace_name, trace, profile, reward_weights, "festive", festive, session_root)
        results.append({"trace": trace_name, "algorithm": "FESTIVE-inspired", **metrics})

        metrics = run_controller(trace_name, trace, profile, reward_weights, "rate_based", rate_based, session_root)
        results.append({"trace": trace_name, "algorithm": "Rate-based", **metrics})

        if ppo_model is not None:
            metrics = run_ppo(trace_name, trace, profile, reward_weights, ppo_model, session_root)
            results.append({"trace": trace_name, "algorithm": "PPO", **metrics})

    result_df = pd.DataFrame(results)
    result_df.to_csv(out_dir / "evaluation_summary.csv", index=False)

    grouped = (
        result_df.groupby("algorithm")[[
            "average_bitrate_mbps",
            "total_rebuffer_s",
            "stall_ratio",
            "switch_count",
            "switch_magnitude_mbps",
            "mean_observed_throughput_mbps",
            "total_qoe",
        ]]
        .mean()
        .sort_values("total_qoe", ascending=False)
    )
    grouped.to_csv(out_dir / "evaluation_grouped.csv")
    plot_algorithm_comparison(out_dir / "evaluation_summary.csv", out_dir / "algorithm_comparison.png")

    leaderboard = result_df.groupby("algorithm")["total_qoe"].agg(["mean", "std", "count"]).sort_values("mean", ascending=False).reset_index()
    leaderboard.to_csv(out_dir / "leaderboard.csv", index=False)

    print(result_df)
    print("\nGrouped summary\n")
    print(grouped)


if __name__ == "__main__":
    main()
