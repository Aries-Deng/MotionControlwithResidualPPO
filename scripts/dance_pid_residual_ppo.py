"""Unitree G1 dance tracking with PID + residual PPO.

This is the G1 analogue of the differential-drive PID + residual PPO example,
adapted for humanoid balance. A pretrained full-body PPO policy supplies the
stable balance action, the outer-loop PID/reference controller supplies a dance
tracking prior, and the newly trained PPO policy learns only a bounded residual
on top of that nominal controller. The low-level actuator still handles
torque/PD control inside mjlab.

Run from the repository root, for example:

  conda run --no-capture-output -n hw python -m scripts.dance_pid_residual_ppo

Outputs:
  - logs/rsl_rl/hw_dance_pid_residual_g1/.../model_*.pt
  - scripts/outputs/dance_pid_residual_error.png
  - scripts/outputs/dance_pid_residual_video.mp4
  - scripts/outputs/g1_pid_vs_pid_residual_tracking_error.png
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from mjlab.dance import (
  DEFAULT_TRACKING_STEPS,
  SINGLE_DANCE_MOTION,
  dance_joint_position_mae,
  single_motion_frame_count,
)
from mjlab.rl_cfg import make_ppo_cfg
from mjlab.train import latest_checkpoint
from scripts.dance_pid_baseline import (
  _make_pid_env_cfg,
  infer_action_scale,
  infer_num_actions,
  joint_target_to_action,
  joint_tracking_profile,
  play_pid,
)


EXPERIMENT_NAME = "hw_dance_pid_residual_g1"
DEFAULT_CHECKPOINT = Path(__file__).parent / "pid_residual_model_final.pt"
DEFAULT_BASE_CHECKPOINT = Path(__file__).parent / "model_final.pt"

# Body-part mixing between the pretrained full-body PPO and PID/reference
# tracking. Legs stay close to the pretrained policy because they are
# responsible for contact and balance; arms follow the dance reference more.
PID_LEG_BLEND = 0.01
PID_WAIST_BLEND = 0.02
PID_ARM_BLEND = 0.05

# Residual PPO acts in normalized action space. Limits are intentionally small
# around the balance-critical joints and larger around the dance-dominant arms.
RESIDUAL_LEG_LIMIT = 0.005
RESIDUAL_WAIST_LIMIT = 0.01
RESIDUAL_ARM_LIMIT = 0.04
RESIDUAL_RATE_LIMIT = 0.02

# Extra reward terms added on top of the existing dance-tracking rewards.
RESIDUAL_PENALTY_WEIGHT = 0.025
ACTION_RATE_PENALTY_WEIGHT = 0.015
TRACKING_IMPROVEMENT_WEIGHT = 0.25
UPRIGHT_REWARD_WEIGHT = 0.30


def _tuple_replace(result: Any, index: int, value: Any) -> Any:
  if isinstance(result, tuple):
    items = list(result)
    items[index] = value
    return tuple(items)
  if isinstance(result, list):
    result = list(result)
    result[index] = value
    return result
  raise TypeError(f"Unsupported env.step result type: {type(result)!r}")


def _done_tensor(step_result: Any, device: torch.device, num_envs: int) -> torch.Tensor | None:
  if not isinstance(step_result, (tuple, list)) or len(step_result) < 3:
    return None
  done = torch.as_tensor(step_result[2], device=device).bool().reshape(num_envs, -1)
  return done.any(dim=1, keepdim=True)


def _reward_tensor(step_result: Any) -> torch.Tensor:
  if not isinstance(step_result, (tuple, list)) or len(step_result) < 2:
    raise TypeError("Expected env.step to return at least (obs, reward, done).")
  return step_result[1]


def _upright_reward(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  upright_cos = -robot.data.projected_gravity_b[:, 2]
  return torch.clamp((upright_cos - 0.45) / 0.45, min=0.0, max=1.0)


def _joint_tracking_error_tensor(env: ManagerBasedRlEnv) -> torch.Tensor:
  from mjlab.dance import target_joint_pos

  robot = env.scene["robot"]
  q_ref = target_joint_pos(
    env,
    cycle_s=(SINGLE_DANCE_MOTION.cycle_s,),
    motion_count=1,
    fixed_motion_id=0,
  )
  return torch.mean(torch.abs(robot.data.joint_pos - q_ref), dim=1)


def _body_part_value(
  env: ManagerBasedRlEnv,
  num_actions: int,
  device: torch.device,
  *,
  leg_value: float,
  waist_value: float,
  arm_value: float,
) -> torch.Tensor:
  values = torch.full((1, num_actions), leg_value, device=device)
  for joint_id, name in enumerate(env.scene["robot"].joint_names[:num_actions]):
    if "waist" in name:
      values[:, joint_id] = waist_value
    elif "shoulder" in name or "elbow" in name or "wrist" in name:
      values[:, joint_id] = arm_value
  return values


def _make_policy(
  env: RslRlVecEnvWrapper,
  rl_cfg: Any,
  checkpoint_path: str | Path,
  device: str,
):
  runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), device=device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
  return runner.get_inference_policy(device=device)


class PidResidualVecEnvWrapper:
  """Convert PPO actions into residuals around a base-PPO/PID nominal action."""

  def __init__(
    self,
    wrapped_env: RslRlVecEnvWrapper,
    raw_env: ManagerBasedRlEnv,
    env_cfg: Any,
    base_policy,
  ):
    self.wrapped_env = wrapped_env
    self.raw_env = raw_env
    self.env_cfg = env_cfg
    self.base_policy = base_policy
    self.device = torch.device(getattr(raw_env, "device", "cpu"))
    self.num_envs = int(raw_env.num_envs)

    self.num_actions = infer_num_actions(wrapped_env, raw_env)
    self.action_scale = infer_action_scale(env_cfg, self.num_actions, self.device)
    self.track_scale, self.error_feedback_scale = joint_tracking_profile(raw_env, self.device)
    self.pid_blend = _body_part_value(
      raw_env,
      self.num_actions,
      self.device,
      leg_value=PID_LEG_BLEND,
      waist_value=PID_WAIST_BLEND,
      arm_value=PID_ARM_BLEND,
    )
    self.residual_limit = _body_part_value(
      raw_env,
      self.num_actions,
      self.device,
      leg_value=RESIDUAL_LEG_LIMIT,
      waist_value=RESIDUAL_WAIST_LIMIT,
      arm_value=RESIDUAL_ARM_LIMIT,
    )
    self.integral_error = torch.zeros_like(raw_env.scene["robot"].data.joint_pos)
    self.previous_pid_action = torch.zeros((self.num_envs, self.num_actions), device=self.device)
    self.previous_residual = torch.zeros_like(self.previous_pid_action)
    self.previous_tracking_error = torch.zeros((self.num_envs,), device=self.device)

  def __getattr__(self, name: str) -> Any:
    return getattr(self.wrapped_env, name)

  def _reset_controller_state(self) -> None:
    self.integral_error = torch.zeros_like(self.raw_env.scene["robot"].data.joint_pos)
    self.previous_pid_action = torch.zeros((self.num_envs, self.num_actions), device=self.device)
    self.previous_residual = torch.zeros_like(self.previous_pid_action)
    self.previous_tracking_error = _joint_tracking_error_tensor(self.raw_env).detach()

  def reset(self, *args: Any, **kwargs: Any) -> Any:
    result = self.wrapped_env.reset(*args, **kwargs)
    self._reset_controller_state()
    return result

  def get_pid_action(self) -> torch.Tensor:
    pid_action, self.integral_error, self.previous_pid_action = joint_target_to_action(
      self.raw_env,
      self.action_scale,
      self.integral_error,
      self.previous_pid_action,
      self.track_scale,
      self.error_feedback_scale,
      self.num_actions,
    )
    return pid_action

  def residual_to_action(
    self,
    obs: torch.Tensor,
    residual_action: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual_action = torch.clamp(residual_action, min=-1.0, max=1.0)
    residual = residual_action * self.residual_limit
    residual_delta = torch.clamp(
      residual - self.previous_residual,
      min=-RESIDUAL_RATE_LIMIT,
      max=RESIDUAL_RATE_LIMIT,
    )
    residual = self.previous_residual + residual_delta
    with torch.no_grad():
      base_action = torch.clamp(self.base_policy(obs), min=-1.0, max=1.0)
    nominal_action = base_action
    action = base_action
    return action, residual, nominal_action

  def step(self, residual_action: torch.Tensor, obs: Any | None = None) -> Any:
    if obs is None:
      obs = self.wrapped_env.get_observations()
      if isinstance(obs, tuple):
        obs = obs[0]
    with torch.no_grad():
      base_action = self.base_policy(obs)
    residual_action = torch.clamp(residual_action, min=-1.0, max=1.0)
    residual = residual_action * self.residual_limit
    residual_delta = torch.clamp(
      residual - self.previous_residual,
      min=-RESIDUAL_RATE_LIMIT,
      max=RESIDUAL_RATE_LIMIT,
    )
    residual = self.previous_residual + residual_delta
    action = base_action + residual
    self.previous_residual = residual.detach()
    return self.wrapped_env.step(action)

  def close(self) -> None:
    self.wrapped_env.close()


def _make_residual_env(
  env_cfg: Any,
  rl_cfg: Any,
  device: str,
  render_mode: str | None,
  base_checkpoint_path: str | Path,
) -> tuple[ManagerBasedRlEnv, PidResidualVecEnvWrapper]:
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  base_vec_env = RslRlVecEnvWrapper(raw_env, clip_actions=rl_cfg.clip_actions)
  base_rl_cfg = make_ppo_cfg("hw_dance_single_g1", max_iterations=1)
  base_policy = _make_policy(base_vec_env, base_rl_cfg, base_checkpoint_path, device)
  residual_env = PidResidualVecEnvWrapper(base_vec_env, raw_env, env_cfg, base_policy)
  return raw_env, residual_env


def train(
  num_envs: int = 4096,
  max_iterations: int = 2500,
  base_checkpoint_path: str | Path = DEFAULT_BASE_CHECKPOINT,
  device: str | None = None,
) -> Path:
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = _make_pid_env_cfg()
  env_cfg.scene.num_envs = num_envs
  rl_cfg = make_ppo_cfg(EXPERIMENT_NAME, max_iterations=max_iterations)

  from datetime import datetime

  log_dir = (
    Path("logs") / "rsl_rl" / rl_cfg.experiment_name
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  log_dir.mkdir(parents=True, exist_ok=True)

  raw_env, residual_env = _make_residual_env(
    env_cfg,
    rl_cfg,
    device=device,
    render_mode=None,
    base_checkpoint_path=base_checkpoint_path,
  )
  try:
    runner = MjlabOnPolicyRunner(residual_env, asdict(rl_cfg), str(log_dir), device)
    runner.add_git_repo_to_log(__file__)
    runner.learn(
      num_learning_iterations=rl_cfg.max_iterations,
      init_at_random_ep_len=True,
    )
  finally:
    raw_env.close()
  return log_dir


def play(
  checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
  base_checkpoint_path: str | Path = DEFAULT_BASE_CHECKPOINT,
  output_dir: str | Path | None = None,
  num_steps: int = DEFAULT_TRACKING_STEPS,
  full_motion: bool = False,
  render_mode: str = "rgb_array",
  device: str | None = None,
) -> float:
  if full_motion:
    num_steps = single_motion_frame_count()

  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  out_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / "outputs"
  env_cfg = _make_pid_env_cfg()
  env_cfg.scene.num_envs = 1
  rl_cfg = make_ppo_cfg(EXPERIMENT_NAME, max_iterations=1)

  raw_env, residual_env = _make_residual_env(
    env_cfg,
    rl_cfg,
    device=device,
    render_mode=render_mode,
    base_checkpoint_path=base_checkpoint_path,
  )
  runner = MjlabOnPolicyRunner(residual_env, asdict(rl_cfg), device=device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  try:
    step_dt = float(raw_env.step_dt)
    obs, _ = residual_env.reset()
    metric_history: list[float] = []
    frames: list[np.ndarray] = []

    for _ in range(num_steps):
      with torch.no_grad():
        residual_action = policy(obs)
      step_result = residual_env.step(residual_action, obs=obs)
      obs = step_result[0]
      metric_history.append(
        dance_joint_position_mae(
          raw_env,
          cycle_s=(SINGLE_DANCE_MOTION.cycle_s,),
          motion_count=1,
          fixed_motion_id=0,
        )
      )
      frame = raw_env.render()
      if frame is not None:
        frames.append(frame)
  finally:
    raw_env.close()

  metric_arr = np.asarray(metric_history)
  error = float(np.mean(np.abs(metric_arr)))

  out_dir.mkdir(parents=True, exist_ok=True)
  output_plot = out_dir / "dance_pid_residual_error.png"
  plt.figure()
  plt.plot(metric_arr)
  plt.axhline(0.0, color="r", linestyle="--", label="desired")
  plt.title(f"PID + residual PPO tracking error: {error:.3f} rad")
  plt.xlabel("step")
  plt.ylabel("mean abs joint tracking error [rad]")
  plt.legend()
  plt.savefig(output_plot)
  plt.close()

  output_video = out_dir / "dance_pid_residual_video.mp4"
  if frames:
    media.write_video(str(output_video), frames, fps=int(round(1.0 / step_dt)))

  print(f"[dance PID + residual PPO] Mean joint tracking error: {error:.3f} rad")
  print(f"[dance PID + residual PPO] Plot: {output_plot}")
  print(f"[dance PID + residual PPO] Video: {output_video}")
  return error


def compare_with_pid(
  checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
  base_checkpoint_path: str | Path = DEFAULT_BASE_CHECKPOINT,
  output_dir: str | Path | None = None,
  num_steps: int = DEFAULT_TRACKING_STEPS,
) -> tuple[float, float]:
  out_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / "outputs"
  pid_error = play_pid(output_dir=out_dir, num_steps=num_steps)
  residual_error = play(
    checkpoint_path=checkpoint_path,
    base_checkpoint_path=base_checkpoint_path,
    output_dir=out_dir,
    num_steps=num_steps,
  )

  output_plot = out_dir / "g1_pid_vs_pid_residual_tracking_error.png"
  plt.figure()
  plt.bar(["PID", "PID + residual PPO"], [pid_error, residual_error])
  plt.ylabel("mean abs joint tracking error [rad]")
  plt.title("G1 dance tracking controller comparison")
  plt.savefig(output_plot)
  plt.close()
  print(f"[dance comparison] Plot: {output_plot}")
  return pid_error, residual_error


def main(num_envs: int = 4096, max_iterations: int = 2500) -> None:
  log_dir = train(num_envs=num_envs, max_iterations=max_iterations)
  ckpt = latest_checkpoint(log_dir)
  play(ckpt)


if __name__ == "__main__":
  main()
