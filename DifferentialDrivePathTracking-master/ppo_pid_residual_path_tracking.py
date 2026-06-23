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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

import ppo_path_tracking
from main import (
    State,
    challenge_keyframes,
    make_time_indexed_trajectory,
    pose_tracking_error,
)


DT = 0.1
LOOKAHEAD_STEPS = 9
MAX_LINEAR_SPEED = 4.0
MAX_ANGULAR_SPEED = 4.5
POSITION_SCALE = 25.0

PID_KP = 2.0
PID_KI = 0.005
PID_KD = 0.25
PID_SPEED_SCALE = 0.985

RESIDUAL_LINEAR_LIMIT = 1.80
RESIDUAL_ANGULAR_LIMIT = 5.00

TURN_SLOWDOWN_WINDOW = 0.5
TURN_SPEED_SCALE = 0.80
BC_PRETRAIN_STEPS = 600
BC_LOSS_COEF = 0.35
PPO_TEACHER_MODEL = _OUT_DIR / "ppo_differential_drive_model.pt"


class ResidualPIDTrackingEnv:
    """PPO learns bounded residual commands on top of a PID trajectory tracker."""

    def __init__(self, num_envs=128, device="cpu", training=True):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.training = training
        self.reference = make_time_indexed_trajectory(challenge_keyframes(), DT)
        self.ref_x = torch.tensor([p.x for p in self.reference], device=self.device, dtype=torch.float32)
        self.ref_y = torch.tensor([p.y for p in self.reference], device=self.device, dtype=torch.float32)
        self.ref_theta = torch.tensor([p.theta for p in self.reference], device=self.device, dtype=torch.float32)
        self.ref_speed = torch.tensor([p.speed for p in self.reference], device=self.device, dtype=torch.float32)
        self.ref_t = torch.tensor([p.t for p in self.reference], device=self.device, dtype=torch.float32)
        self.num_steps = len(self.reference)
        heading_step = torch.atan2(
            torch.sin(self.ref_theta[1:] - self.ref_theta[:-1]),
            torch.cos(self.ref_theta[1:] - self.ref_theta[:-1]),
        )
        turn_indices = torch.nonzero(torch.abs(heading_step) > 0.25, as_tuple=False).flatten() + 1
        self.turn_times = self.ref_t[turn_indices]

        self.x = torch.zeros(num_envs, device=self.device)
        self.y = torch.zeros(num_envs, device=self.device)
        self.theta = torch.zeros(num_envs, device=self.device)
        self.step_idx = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.pid_integral = torch.zeros(num_envs, device=self.device)
        self.pid_prev_error = torch.zeros(num_envs, device=self.device)
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

        self.pid_integral[env_ids] = 0.0
        self.pid_prev_error[env_ids] = 0.0
        return self.obs()

    def _frame(self, offset=0):
        return torch.clamp(self.step_idx + offset, max=self.num_steps - 1)

    def _turn_strength(self, offset=0):
        frame = self._frame(offset)
        if self.turn_times.numel() == 0:
            return torch.zeros(self.num_envs, device=self.device)
        time_to_turn = torch.min(torch.abs(self.ref_t[frame].unsqueeze(1) - self.turn_times.unsqueeze(0)), dim=1).values
        return (time_to_turn <= TURN_SLOWDOWN_WINDOW).float()

    def _pid_command(self):
        frame = self._frame(0)
        target = self._frame(LOOKAHEAD_STEPS)

        dx = self.ref_x[target] - self.x
        dy = self.ref_y[target] - self.y
        goal_theta = torch.atan2(dy, dx)
        heading_error = torch.atan2(
            torch.sin(goal_theta - self.theta),
            torch.cos(goal_theta - self.theta),
        )
        error_integral = self.pid_integral + heading_error
        error_derivative = heading_error - self.pid_prev_error
        yaw_rate = PID_KP * heading_error + PID_KI * error_integral + PID_KD * error_derivative
        yaw_rate = torch.atan2(torch.sin(yaw_rate), torch.cos(yaw_rate))
        yaw_rate = torch.clamp(yaw_rate, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        speed = torch.clamp(self.ref_speed[frame] * PID_SPEED_SCALE, 0.0, MAX_LINEAR_SPEED)
        return speed, yaw_rate, heading_error

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
        pid_speed, pid_yaw_rate, _ = self._pid_command()
        turn_now = self._turn_strength(0)
        turn_preview = torch.maximum(self._turn_strength(LOOKAHEAD_STEPS), self._turn_strength(2 * LOOKAHEAD_STEPS))

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
                pid_speed / MAX_LINEAR_SPEED,
                pid_yaw_rate / MAX_ANGULAR_SPEED,
                turn_now,
                turn_preview,
                torch.sin(2.0 * torch.pi * phase),
                torch.cos(2.0 * torch.pi * phase),
            ),
            dim=1,
        )

    def expert_action(self):
        frame = self._frame(0)
        pid_speed, _, _ = self._pid_command()
        turn_strength = self._turn_strength(0)
        speed_scale = PID_SPEED_SCALE - (PID_SPEED_SCALE - TURN_SPEED_SCALE) * turn_strength
        target_speed = torch.clamp(self.ref_speed[frame] * speed_scale, 0.0, MAX_LINEAR_SPEED)
        speed_action = torch.clamp(
            (target_speed - pid_speed) / RESIDUAL_LINEAR_LIMIT,
            -1.0,
            1.0,
        )
        yaw_action = torch.zeros_like(speed_action)
        return torch.stack((speed_action, yaw_action), dim=1)

    def teacher_residual_action(self, teacher_model):
        frame = self._frame(0)
        pid_speed, pid_yaw_rate, _ = self._pid_command()
        with torch.no_grad():
            teacher_obs = self._teacher_obs()
            teacher_action = torch.clamp(teacher_model.actor(teacher_obs), -1.0, 1.0)

        teacher_speed = torch.clamp(
            self.ref_speed[frame] + ppo_path_tracking.SPEED_CORRECTION_LIMIT * teacher_action[:, 0],
            0.0,
            MAX_LINEAR_SPEED,
        )
        teacher_yaw_rate = torch.clamp(
            ppo_path_tracking.MAX_ANGULAR_SPEED * teacher_action[:, 1],
            -MAX_ANGULAR_SPEED,
            MAX_ANGULAR_SPEED,
        )
        speed_action = torch.clamp((teacher_speed - pid_speed) / RESIDUAL_LINEAR_LIMIT, -1.0, 1.0)
        yaw_action = torch.clamp((teacher_yaw_rate - pid_yaw_rate) / RESIDUAL_ANGULAR_LIMIT, -1.0, 1.0)
        return torch.stack((speed_action, yaw_action), dim=1)

    def _teacher_obs(self):
        current = self._frame(0)
        target = self._frame(ppo_path_tracking.LOOKAHEAD_STEPS)

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
                body_x / ppo_path_tracking.POSITION_SCALE,
                body_y / ppo_path_tracking.POSITION_SCALE,
                body_x_la / ppo_path_tracking.POSITION_SCALE,
                body_y_la / ppo_path_tracking.POSITION_SCALE,
                torch.sin(heading_error),
                torch.cos(heading_error),
                torch.sin(lookahead_heading_error),
                torch.cos(lookahead_heading_error),
                self.ref_speed[current] / ppo_path_tracking.MAX_LINEAR_SPEED,
                torch.sin(2.0 * torch.pi * phase),
                torch.cos(2.0 * torch.pi * phase),
            ),
            dim=1,
        )

    def step(self, action):
        action = torch.clamp(action, -1.0, 1.0)
        pid_speed, pid_yaw_rate, pid_error = self._pid_command()
        residual_speed = RESIDUAL_LINEAR_LIMIT * action[:, 0]
        residual_yaw_rate = RESIDUAL_ANGULAR_LIMIT * action[:, 1]

        speed_cmd = torch.clamp(pid_speed + residual_speed, 0.0, MAX_LINEAR_SPEED)
        yaw_rate_cmd = torch.clamp(pid_yaw_rate + residual_yaw_rate, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)

        pid_only_x = self.x + pid_speed * torch.cos(self.theta) * DT
        pid_only_y = self.y + pid_speed * torch.sin(self.theta) * DT
        pid_only_theta = torch.atan2(
            torch.sin(self.theta + pid_yaw_rate * DT),
            torch.cos(self.theta + pid_yaw_rate * DT),
        )

        self.x = self.x + speed_cmd * torch.cos(self.theta) * DT
        self.y = self.y + speed_cmd * torch.sin(self.theta) * DT
        self.theta = torch.atan2(
            torch.sin(self.theta + yaw_rate_cmd * DT),
            torch.cos(self.theta + yaw_rate_cmd * DT),
        )
        self.pid_integral = self.pid_integral + pid_error
        self.pid_prev_error = pid_error
        self.step_idx = torch.clamp(self.step_idx + 1, max=self.num_steps - 1)

        next_frame = self._frame(0)
        dx = self.ref_x[next_frame] - self.x
        dy = self.ref_y[next_frame] - self.y
        pid_dx = self.ref_x[next_frame] - pid_only_x
        pid_dy = self.ref_y[next_frame] - pid_only_y
        cos_t = torch.cos(self.theta)
        sin_t = torch.sin(self.theta)
        longitudinal_error = cos_t * dx + sin_t * dy
        lateral_error = -sin_t * dx + cos_t * dy
        pos_error = torch.sqrt(dx * dx + dy * dy)
        pid_pos_error = torch.sqrt(pid_dx * pid_dx + pid_dy * pid_dy)
        heading_error = torch.abs(
            torch.atan2(
                torch.sin(self.theta - self.ref_theta[next_frame]),
                torch.cos(self.theta - self.ref_theta[next_frame]),
            )
        )
        pid_heading_error = torch.abs(
            torch.atan2(
                torch.sin(pid_only_theta - self.ref_theta[next_frame]),
                torch.cos(pid_only_theta - self.ref_theta[next_frame]),
            )
        )
        improvement = 1.5 * (pid_pos_error - pos_error) + 0.35 * (pid_heading_error - heading_error)
        residual_penalty = 0.10 * torch.sum(action * action, dim=1)
        reward = (
            2.0 * torch.exp(-2.0 * pos_error)
            + 0.8 * torch.exp(-6.0 * torch.abs(lateral_error))
            + 0.5 * torch.exp(-3.0 * torch.abs(longitudinal_error))
            + 0.5 * torch.exp(-2.5 * heading_error)
            - 0.18 * pos_error
            - 0.05 * heading_error
            + improvement
            - residual_penalty
        )
        done = self.step_idx >= self.num_steps - 1
        return self.obs(), reward, done, {
            "position_error": pos_error,
            "heading_error": heading_error,
            "pid_speed": pid_speed,
            "pid_yaw_rate": pid_yaw_rate,
            "speed_cmd": speed_cmd,
            "yaw_rate_cmd": yaw_rate_cmd,
            "residual_speed": speed_cmd - pid_speed,
            "residual_yaw_rate": yaw_rate_cmd - pid_yaw_rate,
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
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))
        nn.init.zeros_(self.actor[-2].weight)
        nn.init.zeros_(self.actor[-2].bias)

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


def load_teacher_model(device="cpu"):
    if not PPO_TEACHER_MODEL.exists():
        return None
    teacher_env = ppo_path_tracking.DifferentialDriveTrackingEnv(num_envs=1, device=device, training=False)
    teacher_model = ppo_path_tracking.ActorCritic(teacher_env.obs().shape[1], 2).to(device)
    teacher_model.load_state_dict(torch.load(PPO_TEACHER_MODEL, map_location=device))
    teacher_model.eval()
    return teacher_model


def expert_actions(env, teacher_model=None):
    if teacher_model is None:
        return env.expert_action()
    return env.teacher_residual_action(teacher_model)


def pretrain_actor_with_expert(model, env, teacher_model=None, steps=BC_PRETRAIN_STEPS, batch_size=1024, device="cpu"):
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1.0e-3)
    for _ in range(steps):
        env.reset()
        obs = env.obs()
        target_action = expert_actions(env, teacher_model)
        if obs.shape[0] < batch_size:
            repeat_count = int(np.ceil(batch_size / obs.shape[0]))
            obs = obs.repeat((repeat_count, 1))[:batch_size]
            target_action = target_action.repeat((repeat_count, 1))[:batch_size]

        prediction = model.actor(obs)
        loss = torch.mean((prediction - target_action) ** 2)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.actor.parameters(), 0.5)
        optimizer.step()


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

    env = ResidualPIDTrackingEnv(num_envs=num_envs, device=device, training=True)
    obs = env.reset()
    model = ActorCritic(obs.shape[1], 2).to(device)
    teacher_model = load_teacher_model(device=device)
    if teacher_model is None:
        print("[PID+PPO residual] PPO teacher model not found; using turn-slowdown warm start.")
    else:
        print(f"[PID+PPO residual] Using PPO teacher: {PPO_TEACHER_MODEL}")
    pretrain_actor_with_expert(model, env, teacher_model=teacher_model, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)

    reward_curve = []
    for iteration in range(1, iterations + 1):
        obs_buf = []
        action_buf = []
        logprob_buf = []
        reward_buf = []
        done_buf = []
        value_buf = []
        expert_action_buf = []

        for _ in range(rollout_steps):
            with torch.no_grad():
                action, log_prob, value = model.act(obs)
            next_obs, reward, done, _ = env.step(action)

            obs_buf.append(obs)
            action_buf.append(action)
            expert_action_buf.append(expert_actions(env, teacher_model))
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
        expert_action_t = torch.stack(expert_action_buf)
        advantages, returns = compute_gae(reward_t, done_t, value_t, last_value)

        flat_obs = obs_t.reshape(-1, obs_t.shape[-1])
        flat_action = action_t.reshape(-1, action_t.shape[-1])
        flat_old_logprob = old_logprob_t.reshape(-1)
        flat_advantage = advantages.reshape(-1)
        flat_return = returns.reshape(-1)
        flat_expert_action = expert_action_t.reshape(-1, expert_action_t.shape[-1])
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
                bc_loss = torch.mean((model.actor(flat_obs[idx]) - flat_expert_action[idx]) ** 2)
                loss = policy_loss + 0.5 * value_loss + BC_LOSS_COEF * bc_loss - 0.01 * entropy

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optimizer.step()

        mean_reward = float(reward_t.mean().detach().cpu())
        reward_curve.append(mean_reward)
        if iteration == 1 or iteration % 10 == 0:
            print(f"[PID+PPO residual] iter={iteration:04d} mean_step_reward={mean_reward:.3f}")

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = _OUT_DIR / "ppo_pid_residual_differential_drive_model.pt"
    torch.save(model.state_dict(), model_path)
    save_reward_curve(reward_curve, _OUT_DIR / "ppo_pid_residual_training_reward.png")
    return model, model_path


def save_reward_curve(reward_curve, output_path):
    plt.figure(figsize=(9, 5))
    plt.plot(reward_curve)
    plt.title("PID + PPO residual training reward")
    plt.xlabel("iteration")
    plt.ylabel("mean step reward")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def evaluate(model, device="cpu"):
    env = ResidualPIDTrackingEnv(num_envs=1, device=device, training=False)
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
        "pid_speed": [],
        "pid_yaw_rate": [],
        "speed_cmd": [],
        "yaw_rate_cmd": [],
        "residual_speed": [],
        "residual_yaw_rate": [],
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
        obs, _, done, info = env.step(action)
        for key in (
                "pid_speed",
                "pid_yaw_rate",
                "speed_cmd",
                "yaw_rate_cmd",
                "residual_speed",
                "residual_yaw_rate"):
            history[key].append(float(info[key][0].detach().cpu()))
        if bool(done[0]):
            break

    for key, value in history.items():
        history[key] = np.asarray(value)

    summary = save_residual_tracking_plots(history, _OUT_DIR)
    print(
        "PID + PPO residual tracking error: "
        f"position={summary['mean_position_error']:.3f} m, "
        f"heading={summary['mean_heading_error']:.3f} rad, "
        f"pose_mae={summary['mean_pose_mae']:.3f}"
    )
    return summary


def save_residual_tracking_plots(history, output_dir):
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
        "PID + PPO residual tracking error: "
        f"pos={mean_position_error:.3f} m, heading={mean_heading_error:.3f} rad"
    )
    plt.xlabel("time [s]")
    plt.ylabel("error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    error_plot = output_dir / "ppo_pid_residual_tracking_error.png"
    plt.savefig(error_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(history["ref_x"], history["ref_y"], "k--", label="time-indexed reference")
    plt.plot(history["x"], history["y"], "b", label="PID + PPO residual rollout")
    plt.scatter(history["ref_x"][0], history["ref_y"][0], c="g", label="start")
    plt.scatter(history["ref_x"][-1], history["ref_y"][-1], c="r", label="end")
    plt.title(f"PID + PPO residual path tracking, pose MAE={mean_pose_mae:.3f}")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path_plot = output_dir / "ppo_pid_residual_tracking_path.png"
    plt.savefig(path_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(history["t"], np.unwrap(history["ref_theta"]), "k--", label="reference heading")
    plt.plot(history["t"], np.unwrap(history["theta"]), "b", label="PID + PPO residual heading")
    plt.title(f"PID + PPO residual heading tracking, mean error={mean_heading_error:.3f} rad")
    plt.xlabel("time [s]")
    plt.ylabel("heading [rad]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    heading_plot = output_dir / "ppo_pid_residual_heading_tracking.png"
    plt.savefig(heading_plot, dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.subplot(2, 1, 1)
    plt.plot(history["t"], history["pid_speed"], "k--", label="PID speed")
    plt.plot(history["t"], history["speed_cmd"], "b", label="final speed")
    plt.plot(history["t"], history["residual_speed"], "g", label="PPO speed residual")
    plt.ylabel("linear speed [m/s]")
    plt.grid(True)
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(history["t"], history["pid_yaw_rate"], "k--", label="PID yaw rate")
    plt.plot(history["t"], history["yaw_rate_cmd"], "b", label="final yaw rate")
    plt.plot(history["t"], history["residual_yaw_rate"], "g", label="PPO yaw residual")
    plt.xlabel("time [s]")
    plt.ylabel("yaw rate [rad/s]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    command_plot = output_dir / "ppo_pid_residual_commands.png"
    plt.savefig(command_plot, dpi=160)
    plt.close()

    return {
        "mean_position_error": mean_position_error,
        "mean_heading_error": mean_heading_error,
        "mean_pose_mae": mean_pose_mae,
        "error_plot": error_plot,
        "path_plot": path_plot,
        "heading_plot": heading_plot,
        "command_plot": command_plot,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_path = train(device=device)
    print(f"Saved PID + PPO residual model: {model_path}")
    summary = evaluate(model, device=device)
    print(f"Saved error plot: {summary['error_plot']}")
    print(f"Saved path plot: {summary['path_plot']}")
    print(f"Saved heading plot: {summary['heading_plot']}")
    print(f"Saved command plot: {summary['command_plot']}")


if __name__ == "__main__":
    main()
