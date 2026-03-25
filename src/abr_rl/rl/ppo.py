from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from abr_rl.config import ExperimentConfig
from abr_rl.envs.abr_env import ABRStreamingEnv
from abr_rl.rl.buffer import RolloutBuffer
from abr_rl.rl.models import ActorCritic
from abr_rl.utils.seed import set_seed


class PPOTrainer:
    def __init__(self, env_fn: Callable[[], ABRStreamingEnv], config: ExperimentConfig):
        self.env_fn = env_fn
        self.cfg = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_num_threads(1)
        set_seed(config.seed)

        env = env_fn()
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        self.model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=config.training.hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.training.learning_rate)
        self.buffer = RolloutBuffer()
        self.history: List[Dict] = []

    def _policy(self, obs: np.ndarray) -> tuple[int, float, float]:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.model(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(value.item())

    def _value(self, obs: np.ndarray) -> float:
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, value = self.model(obs_t)
        return float(value.item())

    def _compute_gae(self, rewards, dones, values, next_value: float):
        gamma = self.cfg.training.gamma
        lam = self.cfg.training.gae_lambda

        advantages = np.zeros_like(rewards, dtype=np.float32)
        lastgaelam = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            lastgaelam = delta + gamma * lam * next_non_terminal * lastgaelam
            advantages[t] = lastgaelam

        returns = advantages + values
        return advantages, returns

    def _update(self, last_obs: np.ndarray, update_idx: int) -> Dict[str, float]:
        data = self.buffer.to_tensors(self.device)
        next_value = 0.0 if bool(data["dones"][-1].item()) else self._value(last_obs)

        advantages, returns = self._compute_gae(
            rewards=data["rewards"].cpu().numpy(),
            dones=data["dones"].cpu().numpy(),
            values=data["values"].cpu().numpy(),
            next_value=next_value,
        )

        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        obs = data["obs"]
        actions = data["actions"]
        old_log_probs = data["log_probs"]

        batch_size = obs.shape[0]
        minibatch_size = min(self.cfg.training.minibatch_size, batch_size)

        policy_loss_total = 0.0
        value_loss_total = 0.0
        entropy_total = 0.0
        updates = 0

        for _ in range(self.cfg.training.update_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_idx = indices[start:start + minibatch_size]
                logits, values = self.model(obs[mb_idx])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()

                ratio = (new_log_probs - old_log_probs[mb_idx]).exp()
                surr1 = ratio * adv_t[mb_idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.cfg.training.clip_range,
                    1.0 + self.cfg.training.clip_range,
                ) * adv_t[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (ret_t[mb_idx] - values).pow(2).mean()
                loss = (
                    policy_loss
                    + self.cfg.training.value_coef * value_loss
                    - self.cfg.training.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.max_grad_norm)
                self.optimizer.step()

                policy_loss_total += float(policy_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy.item())
                updates += 1

        metrics = {
            "update": update_idx,
            "policy_loss": policy_loss_total / max(updates, 1),
            "value_loss": value_loss_total / max(updates, 1),
            "entropy": entropy_total / max(updates, 1),
            "batch_reward_mean": float(np.mean(self.buffer.rewards)) if self.buffer.rewards else 0.0,
        }
        self.buffer.clear()
        return metrics

    def train(self) -> tuple[Path, pd.DataFrame]:
        out_dir = Path(self.cfg.output.run_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = out_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        env = self.env_fn()
        obs, reset_info = env.reset()
        episode_reward = 0.0
        completed_episode_rewards: List[float] = []
        completed_episode_qoe: List[float] = []
        completed_episode_rebuffer: List[float] = []
        completed_episode_bitrate: List[float] = []
        total_steps = 0
        update_idx = 0

        while total_steps < self.cfg.training.total_timesteps:
            self.buffer.clear()
            while len(self.buffer) < self.cfg.training.timesteps_per_batch and total_steps < self.cfg.training.total_timesteps:
                action, log_prob, value = self._policy(obs)
                next_obs, reward, done, truncated, info = env.step(action)
                terminal = done or truncated
                self.buffer.add(obs, action, reward, terminal, log_prob, value)

                episode_reward += reward
                total_steps += 1
                obs = next_obs

                if terminal:
                    completed_episode_rewards.append(float(episode_reward))
                    if "episode_metrics" in info:
                        metrics = info["episode_metrics"]
                        completed_episode_qoe.append(float(metrics.get("total_qoe", 0.0)))
                        completed_episode_rebuffer.append(float(metrics.get("total_rebuffer_s", 0.0)))
                        completed_episode_bitrate.append(float(metrics.get("average_bitrate_mbps", 0.0)))
                    episode_reward = 0.0
                    obs, reset_info = env.reset()

            update_idx += 1
            metrics = self._update(obs, update_idx=update_idx)
            metrics["total_steps"] = total_steps
            metrics["episodes_finished"] = len(completed_episode_rewards)
            metrics["episode_reward_mean"] = float(np.mean(completed_episode_rewards[-10:])) if completed_episode_rewards else 0.0
            metrics["train_qoe_mean_last10"] = float(np.mean(completed_episode_qoe[-10:])) if completed_episode_qoe else 0.0
            metrics["train_rebuffer_mean_last10"] = float(np.mean(completed_episode_rebuffer[-10:])) if completed_episode_rebuffer else 0.0
            metrics["train_bitrate_mean_last10"] = float(np.mean(completed_episode_bitrate[-10:])) if completed_episode_bitrate else 0.0
            self.history.append(metrics)

            print(
                f"Update {update_idx}: steps={total_steps}, "
                f"episode_reward_mean={metrics['episode_reward_mean']:.3f}, "
                f"train_qoe_mean_last10={metrics['train_qoe_mean_last10']:.3f}, "
                f"policy_loss={metrics['policy_loss']:.4f}, "
                f"value_loss={metrics['value_loss']:.4f}",
                flush=True,
            )

            if update_idx % self.cfg.training.checkpoint_every_updates == 0 or total_steps >= self.cfg.training.total_timesteps:
                checkpoint_path = checkpoint_dir / f"ppo_update_{update_idx:03d}.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "config": self.cfg.to_dict(),
                        "history": self.history,
                    },
                    checkpoint_path,
                )

        history_df = pd.DataFrame(self.history)
        history_df.to_csv(out_dir / "training_history.csv", index=False)

        final_path = out_dir / "ppo_final.pt"
        torch.save({"model_state_dict": self.model.state_dict()}, final_path)
        return final_path, history_df



def load_model(model_path: str | Path, obs_dim: int, action_dim: int, hidden_dim: int, device: torch.device | None = None) -> ActorCritic:
    device = device or torch.device("cpu")
    model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model



def ppo_action(model: ActorCritic, obs: np.ndarray, device: torch.device | None = None) -> int:
    device = device or next(model.parameters()).device
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs_t)
        return int(torch.argmax(logits, dim=-1).item())
