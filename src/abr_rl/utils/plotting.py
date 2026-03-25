from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd



def plot_training_history(history_csv: str | Path, output_path: str | Path) -> None:
    history = pd.read_csv(history_csv)
    if history.empty:
        return

    plt.figure(figsize=(9, 5))
    plt.plot(history["total_steps"], history["episode_reward_mean"], label="Episode reward mean")
    if "train_qoe_mean_last10" in history.columns:
        plt.plot(history["total_steps"], history["train_qoe_mean_last10"], label="QoE mean (last 10 episodes)")
    plt.xlabel("Total steps")
    plt.ylabel("Score")
    plt.title("PPO training progress")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()



def plot_algorithm_comparison(summary_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(summary_csv)
    if df.empty:
        return

    grouped = df.groupby("algorithm")["total_qoe"].mean().sort_values(ascending=False)

    plt.figure(figsize=(8, 4.5))
    grouped.plot(kind="bar")
    plt.ylabel("Mean total QoE")
    plt.title("Algorithm comparison")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()



def plot_trace_examples(catalog_csv: str | Path, output_path: str | Path) -> None:
    catalog = pd.read_csv(catalog_csv)
    if catalog.empty:
        return
    grouped = catalog.groupby("scenario")["mean_mbps"].mean().sort_values(ascending=False)
    plt.figure(figsize=(9, 4.5))
    grouped.plot(kind="bar")
    plt.ylabel("Mean bandwidth (Mbps)")
    plt.title("Scenario-level average bandwidth")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)
    plt.close()
