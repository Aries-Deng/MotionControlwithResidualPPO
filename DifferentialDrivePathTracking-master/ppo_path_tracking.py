from pathlib import Path
import os
import sys

_OUT_DIR = Path(__file__).resolve().parent / "outputs"
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
_CACHE_DIR = _OUT_DIR / ".cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from main import (
    State,
    challenge_keyframes,
    fixAngle,
    make_time_indexed_trajectory,
    pose_tracking_error,
)


DT = 0.1
LOOKAHEAD_STEPS = 10
MAX_LINEAR_SPEED = 4.0
MAX_ANGULAR_SPEED = 4.5
POSITION_SCALE = 25.0
SPEED_CORRECTION_LIMIT = 1.5


class DifferentialDriveTrackingEnv:
    """Vectorized time-indexed trajectory-tracking environment."""

    def __init__(self, num_envs=128, device="cpu", training=True):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.training = training
        self.reference = make_time_indexed_trajectory(challenge_keyframes(), DT)
        self.ref_x = torch.tensor([p.x for p in self.reference], device=self.device, dtype=torch.float32)
        self.ref_y = torch.tensor([p.y for p in self.reference], device=self.device, dtype=torch.float32)
        self.ref_theta = torch.tensor([p.theta for p in self.reference], device=self.device, dtype=torch.float32)
        self.ref_speed = torch.tensor([p.speed for p in self.reference], device=self.device, dtype=torch.float32)
        self.num_steps = len(self.reference)

        self.x = torch.zeros(num_envs, device=self.device)
        self.y = torch.zeros(num_envs, device=self.device)
        self.theta = torch.zeros(num_envs, device=self.device)
        self.step_idx = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.reset()

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        elif not torch.is_tensor(env_ids):
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)

        if self.training:
            max_start = max(self.num_steps - LOOKAHEAD_STEPS - 2, 1)
            start_idx = torch.randint(0, max_start, (len(env_ids),), device=self.device)
        else:
            start_idx = torch.zeros(len(env_ids), device=self.device, dtype=torch.long)

        self.step_idx[env_ids] = start_idx
        self.x[env_ids] = self.ref_x[start_idx]
        self.y[env_ids] = self.ref_y[start_idx]
        self.theta[env_ids] = self.ref_theta[start_idx]
        if not self.training:
            self.theta[env_ids] = float(np.radians(90))

        if self.training:
            self.x[env_ids] += torch.empty(len(env_ids), device=self.device).uniform_(-0.35, 0.35)
            self.y[env_ids] += torch.empty(len(env_ids), device=self.device).uniform_(-0.35, 0.35)
            self.theta[env_ids] += torch.empty(len(env_ids), device=self.device).uniform_(-0.20, 0.20)
        return self.obs()

    def _frame(self, offset=0):
        return torch.clamp(self.step_idx + offset, max=self.num_steps - 1)

    def obs(self):
        current = self._frame(0)
        target = self._frame(LOOKAHEAD_STEPS)

        dx = self.ref_x[current] - self.x
        dy = self.ref_y[current] - self.y
        dx_la = self.ref_x[target] - self.x
        dy_la = self.ref_y[target] - self.y

        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        body_x = cos_t * dx + sin_t * dy
        body_y = -sin_t * dx + cos_t * dy
        body_x_la = cos_t * dx_la + sin_t * dy_la
        body_y_la = -sin_t * dx_la + cos_t * dy_la

        heading_error = torch.atan2(
            torch.sin(self.ref_theta[current] - self.theta),
            torch.cos(self.ref_theta[current] - self.theta),
        )
        lookahead_heading_error = torch.atan2(
            torch.sin(self.ref_theta[target] - self.theta),
            torch.cos(self.ref_theta[target] - self.theta),
        )
        phase = self.step_idx.float() / max(float(self.num_steps - 1), 1.0)

        return torch.stack(
            (
                body_x / POSITION_SCALE,
                body_y / POSITION_SCALE,
                body_x_la / POSITION_SCALE,
                body_y_la / POSITION_SCALE,
                torch.sin(heading_error),
                torch.cos(heading_error),
                torch.sin(lookahead_heading_error),
                torch.cos(lookahead_heading_error),
                self.ref_speed[current] / MAX_LINEAR_SPEED,
                torch.sin(2.0 * torch.pi * phase),
                torch.cos(2.0 * torch.pi * phase),
            ),
            dim=1,
        )

    def step(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        frame = self._frame(0)
        speed_cmd = torch.clamp(self.ref_speed[frame] + SPEED_CORRECTION_LIMIT * action[:, 0], 0.0, MAX_LINEAR_SPEED)
        yaw_rate_cmd = MAX_ANGULAR_SPEED * action[:, 1]

        self.x = self.x + speed_cmd * torch.cos(self.theta) * DT
        self.y = self.y + speed_cmd * torch.sin(self.theta) * DT
        self.theta = torch.atan2(
            torch.sin(self.theta + yaw_rate_cmd * DT),
            torch.cos(self.theta + yaw_rate_cmd * DT),
        )
        self.step_idx = torch.clamp(self.step_idx + 1, max=self.num_steps - 1)

        next_frame = self._frame(0)
        dx = self.ref_x[next_frame] - self.x
        dy = self.ref_y[next_frame] - self.y
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        longitudinal_error = cos_t * dx + sin_t * dy
        lateral_error = -sin_t * dx + cos_t * dy
        pos_error = torch.sqrt(dx * dx + dy * dy)
        heading_error = torch.abs(
            torch.atan2(
                torch.sin(self.theta - self.ref_theta[next_frame]),
                torch.cos(self.theta - self.ref_theta[next_frame]),
            )
        )
        action_penalty = 0.01 * torch.sum(action * action, dim=1)
        reward = (
            2.0 * torch.exp(-2.0 * pos_error)
            + 0.7 * torch.exp(-6.0 * torch.abs(lateral_error))
            + 0.5 * torch.exp(-3.0 * torch.abs(longitudinal_error))
            + 0.5 * torch.exp(-2.5 * heading_error)
            - 0.18 * pos_error
            - 0.05 * heading_error
            - action_penalty
        )
        done = self.step_idx >= self.num_steps - 1
        return self.obs(), reward, done, {
            "position_error": pos_error,
            "heading_error": heading_error,
        }


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.8))

    def distribution(self, obs):
        mean = self.actor(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs):
        dist = self.distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        return action, log_prob, self.value(obs)


def compute_gae(rewards, dones, values, last_value, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    next_advantage = torch.zeros(rewards.shape[1], device=rewards.device)
    next_value = last_value
    for t in reversed(range(rewards.shape[0])):
        not_done = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_advantage = delta + gamma * lam * not_done * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    returns = advantages + values
    return advantages, returns


def train(
        num_envs=128,
        rollout_steps=128,
        iterations=200,
        minibatch_size=1024,
        update_epochs=5,
        seed=0,
        device="cpu"):
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = DifferentialDriveTrackingEnv(num_envs=num_envs, device=device, training=True)
    obs = env.reset()
    model = ActorCritic(obs.shape[1], 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)

    reward_curve = []
    for iteration in range(1, iterations + 1):
        obs_buf = []
        action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []

        for _ in range(rollout_steps):
            with torch.no_grad():
                action, log_prob, value = model.act(obs)
            next_obs, reward, done, _ = env.step(action)

            obs_buf.append(obs)
            action_buf.append(action)
            logprob_buf.append(log_prob)
            reward_buf.append(reward)
            done_buf.append(done)
            value_buf.append(value)

            obs = next_obs
            if done.any():
                env.reset(torch.nonzero(done, as_tuple=False).flatten())
                obs = env.obs()

        with torch.no_grad():
            last_value = model.value(obs)

        obs_t = torch.stack(obs_buf)
        action_t = torch.stack(action_buf)
        old_logprob_t = torch.stack(logprob_buf)
        reward_t = torch.stack(reward_buf)
        done_t = torch.stack(done_buf)
        value_t = torch.stack(value_buf)
        advantages, returns = compute_gae(reward_t, done_t, value_t, last_value)

        flat_obs = obs_t.reshape(-1, obs_t.shape[-1])
        flat_action = action_t.reshape(-1, action_t.shape[-1])
        flat_old_logprob = old_logprob_t.reshape(-1)
        flat_advantage = advantages.reshape(-1)
        flat_return = returns.reshape(-1)
        flat_advantage = (flat_advantage - flat_advantage.mean()) / (flat_advantage.std() + 1.0e-8)

        total_samples = flat_obs.shape[0]
        for _ in range(update_epochs):
            permutation = torch.randperm(total_samples, device=flat_obs.device)
            for start in range(0, total_samples, minibatch_size):
                idx = permutation[start:start + minibatch_size]
                dist = model.distribution(flat_obs[idx])
                new_logprob = dist.log_prob(flat_action[idx]).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()
                ratio = torch.exp(new_logprob - flat_old_logprob[idx])

                unclipped = ratio * flat_advantage[idx]
                clipped = torch.clamp(ratio, 0.8, 1.2) * flat_advantage[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value = model.value(flat_obs[idx])
                value_loss = torch.mean((value - flat_return[idx]) ** 2)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        mean_reward = float(reward_t.mean().detach().cpu())
        reward_curve.append(mean_reward)
        if iteration == 1 or iteration % 10 == 0:
            print(f"[PPO] iter={iteration:04d} mean_step_reward={mean_reward:.3f}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _OUT_DIR / "ppo_differential_drive_model.pt"
    torch.save(model.state_dict(), model_path)
    save_reward_curve(reward_curve, _OUT_DIR / "ppo_training_reward.png")
    return model, model_path


def save_reward_curve(reward_curve, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(reward_curve)
    plt.title("PPO differential-drive training reward")
    plt.xlabel("iteration")
    plt.ylabel("mean step reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate(model, device="cpu"):
    env = DifferentialDriveTrackingEnv(num_envs=1, device=device, training=False)
    obs = env.reset()
    history = {
        "t": [],
        "x": [],
        "y": [],
        "theta": [],
        "ref_x": [],
        "ref_y": [],
        "ref_theta": [],
        "position_error": [],
        "heading_error": [],
        "pose_mae": [],
    }

    for _ in range(env.num_steps):
        idx = int(env.step_idx[0].item())
        ref = env.reference[idx]
        actual = State(float(env.x[0].item()), float(env.y[0].item()), float(env.theta[0].item()))
        pos_err, head_err, pose_mae = pose_tracking_error(actual, ref)

        history["t"].append(ref.t)
        history["x"].append(actual.x)
        history["y"].append(actual.y)
        history["theta"].append(actual.theta)
        history["ref_x"].append(ref.x)
        history["ref_y"].append(ref.y)
        history["ref_theta"].append(ref.theta)
        history["position_error"].append(pos_err)
        history["heading_error"].append(head_err)
        history["pose_mae"].append(pose_mae)

        with torch.no_grad():
            action = model.actor(obs)
        obs, _, done, _ = env.step(action)
        if bool(done[0]):
            break

    for key, value in history.items():
        history[key] = np.asarray(value)

    summary = save_ppo_tracking_plots(history, _OUT_DIR)
    print(
        "PPO differential-drive tracking error: "
        f"position={summary['mean_position_error']:.3f} m, "
        f"heading={summary['mean_heading_error']:.3f} rad, "
        f"pose_mae={summary['mean_pose_mae']:.3f}"
    )
    return summary


def save_ppo_tracking_plots(history, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mean_position_error = float(np.mean(history["position_error"]))
    mean_heading_error = float(np.mean(history["heading_error"]))
    mean_pose_mae = float(np.mean(history["pose_mae"]))

    plt.figure(figsize=(9, 5))
    plt.plot(history["t"], history["position_error"], label="position error [m]")
    plt.plot(history["t"], history["heading_error"], label="heading error [rad]")
    plt.axhline(0.0, color="r", linestyle="--", label="desired")
    plt.title(
        "PPO differential-drive tracking error: "
        f"pos={mean_position_error:.3f} m, heading={mean_heading_error:.3f} rad"
    )
    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    error_plot = output_dir / "ppo_differential_drive_tracking_error.png"
    plt.savefig(error_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(history["ref_x"], history["ref_y"], "k--", label="time-indexed reference")
    plt.plot(history["x"], history["y"], "b", label="PPO rollout")
    plt.scatter(history["ref_x"][0], history["ref_y"][0], c="g", label="start")
    plt.scatter(history["ref_x"][-1], history["ref_y"][-1], c="r", label="end")
    plt.title(f"PPO differential-drive path tracking, pose MAE={mean_pose_mae:.3f}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path_plot = output_dir / "ppo_differential_drive_tracking_path.png"
    plt.savefig(path_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history["t"], np.unwrap(history["ref_theta"]), "k--", label="reference heading")
    plt.plot(history["t"], np.unwrap(history["theta"]), "b", label="PPO heading")
    plt.title(f"PPO heading tracking, mean error={mean_heading_error:.3f} rad")
    plt.xlabel("time [s]")
    plt.ylabel("heading [rad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    heading_plot = output_dir / "ppo_differential_drive_heading_tracking.png"
    plt.savefig(heading_plot, dpi=160)
    plt.close()

    return {
        "mean_position_error": mean_position_error,
        "mean_heading_error": mean_heading_error,
        "mean_pose_mae": mean_pose_mae,
        "error_plot": error_plot,
        "path_plot": path_plot,
        "heading_plot": heading_plot,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_path = train(device=device)
    print(f"Saved PPO model: {model_path}")
    evaluate(model, device=device)


if __name__ == "__main__":
    main()
