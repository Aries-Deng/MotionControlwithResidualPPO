"""Part 3 baseline - reference/PID-style dance tracking without PPO.

This script uses the same Unitree G1 robot, the same mjlab dance-tracking
environment, and the same hiphop reference trajectory as ``part3_dance_single``.
Instead of loading a learned policy, it converts the reference joint trajectory
into actions directly. In the common mjlab/IsaacLab-style G1 setup, actions are
normalized joint-position offsets and the low-level actuator is already a PD
servo, so this is best read as an outer-loop PID/PD reference-tracking baseline.

Run from the homework repository root, for example:

  conda run --no-capture-output -n mjlab python hw_py/part3_dance_pid_baseline.py
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

from hw_mjlab.dance import (
  DEFAULT_TRACKING_STEPS,
  SINGLE_DANCE_MOTION,
  dance_joint_position_mae,
  single_motion_frame_count,
  target_joint_pos,
  target_joint_vel,
)
from hw_py.part3_dance_single import _make_tracking_env_cfg
from hw_mjlab.rl_cfg import make_ppo_cfg


# If mjlab's action config cannot be inspected, this fallback is used. Override
# it after printing cfg.actions if your local G1 task uses a different scale.
ACTION_SCALE_OVERRIDE: float | None = None
ACTION_SCALE_FALLBACK = 0.25

# Outer-loop PID gains in "position target" space. The inner actuator still owns
# the real torque-level PD control.
KP_TARGET = 0.30
KD_TO_POSITION_SECONDS = 0.04
KI_TO_POSITION_SECONDS = 0.0
INTEGRAL_LIMIT_RAD_S = 0.5
TARGET_DELTA_LIMIT_RAD = 2.2

# Stability-first tracking profile. The reference motion is not guaranteed to be
# dynamically feasible when replayed open-loop, so the baseline tracks arms more
# strongly than legs and backs off when the torso starts falling.
LEG_TRACK_SCALE = 0.10
WAIST_TRACK_SCALE = 0.30
ARM_TRACK_SCALE = 5.00
ARM_ERROR_FEEDBACK = 1.20
ACTION_SMOOTHING = 0.95
ACTION_RATE_LIMIT = 0.60
MIN_UPRIGHT_COS = 0.50
FULL_UPRIGHT_COS = 0.85
MIN_STABLE_HEIGHT = 0.45
FULL_STABLE_HEIGHT = 0.68


def _make_pid_env_cfg():
  env_cfg = _make_tracking_env_cfg(play=True)
  env_cfg.scene.num_envs = 1
  return env_cfg


def _first_attr(obj: Any, names: tuple[str, ...]) -> Any | None:
  for name in names:
    if hasattr(obj, name):
      return getattr(obj, name)
  return None


def _iter_action_terms(actions: Any) -> list[Any]:
  if actions is None:
    return []
  if isinstance(actions, dict):
    return list(actions.values())
  if hasattr(actions, "__dict__"):
    return [
      value for name, value in vars(actions).items()
      if not name.startswith("_") and value is not None
    ]
  return [actions]


def _as_action_scale(value: Any, num_actions: int, device: torch.device) -> torch.Tensor | None:
  if value is None:
    return None
  if isinstance(value, dict):
    values = list(value.values())
    if not values:
      return None
    if all(isinstance(item, (int, float)) for item in values):
      tensor = torch.tensor(values, device=device, dtype=torch.float32)
      if tensor.numel() == num_actions:
        return tensor.reshape(1, -1)
      if tensor.numel() == 1:
        return tensor.reshape(1, 1).expand(1, num_actions)
    return None
  if isinstance(value, (int, float)):
    return torch.full((1, num_actions), float(value), device=device)
  if isinstance(value, (list, tuple, np.ndarray)):
    tensor = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(1, -1)
    if tensor.numel() == 1:
      return tensor.expand(1, num_actions)
    if tensor.shape[1] == num_actions:
      return tensor
  return None


def infer_action_scale(env_cfg: Any, num_actions: int, device: torch.device) -> torch.Tensor:
  """Infer normalized joint-position action scale from cfg.actions."""
  if ACTION_SCALE_OVERRIDE is not None:
    return torch.full((1, num_actions), float(ACTION_SCALE_OVERRIDE), device=device)

  for term in _iter_action_terms(getattr(env_cfg, "actions", None)):
    scale = _first_attr(term, ("scale", "action_scale"))
    parsed = _as_action_scale(scale, num_actions, device)
    if parsed is not None:
      return parsed

    params = getattr(term, "params", None)
    if isinstance(params, dict):
      parsed = _as_action_scale(params.get("scale") or params.get("action_scale"), num_actions, device)
      if parsed is not None:
        return parsed

  print(
    "[PID baseline] Could not infer cfg.actions scale; "
    f"using fallback scale={ACTION_SCALE_FALLBACK}."
  )
  return torch.full((1, num_actions), ACTION_SCALE_FALLBACK, device=device)


def infer_num_actions(wrapped_env: Any, raw_env: Any) -> int:
  for obj in (wrapped_env, raw_env):
    for name in ("num_actions", "action_dim"):
      value = getattr(obj, name, None)
      if value is not None:
        return int(value)
    action_space = getattr(obj, "single_action_space", None) or getattr(obj, "action_space", None)
    shape = getattr(action_space, "shape", None)
    if shape:
      return int(np.prod(shape))
  raise AttributeError("Could not infer action dimension from the mjlab environment.")


def joint_tracking_profile(env: ManagerBasedRlEnv, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
  robot = env.scene["robot"]
  scales: list[float] = []
  error_feedback: list[float] = []
  for name in robot.joint_names:
    if any(part in name for part in ("hip", "knee", "ankle")):
      scales.append(LEG_TRACK_SCALE)
      error_feedback.append(0.0)
    elif "waist" in name:
      scales.append(WAIST_TRACK_SCALE)
      error_feedback.append(0.0)
    else:
      scales.append(ARM_TRACK_SCALE)
      error_feedback.append(ARM_ERROR_FEEDBACK)
  return (
    torch.tensor(scales, device=device, dtype=torch.float32).reshape(1, -1),
    torch.tensor(error_feedback, device=device, dtype=torch.float32).reshape(1, -1),
  )


def stability_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
  robot = env.scene["robot"]
  upright_cos = -robot.data.projected_gravity_b[:, 2:3]
  upright_scale = torch.clamp(
    (upright_cos - MIN_UPRIGHT_COS) / max(FULL_UPRIGHT_COS - MIN_UPRIGHT_COS, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  height = robot.data.root_link_pos_w[:, 2:3]
  height_scale = torch.clamp(
    (height - MIN_STABLE_HEIGHT) / max(FULL_STABLE_HEIGHT - MIN_STABLE_HEIGHT, 1.0e-6),
    min=0.0,
    max=1.0,
  )
  return torch.minimum(upright_scale, height_scale)


def joint_target_to_action(
  env: ManagerBasedRlEnv,
  action_scale: torch.Tensor,
  integral_error: torch.Tensor,
  previous_action: torch.Tensor,
  track_scale: torch.Tensor,
  error_feedback_scale: torch.Tensor,
  num_actions: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Convert reference joint targets into normalized position-offset actions."""
  robot = env.scene["robot"]
  q_ref = target_joint_pos(
    env,
    cycle_s=(SINGLE_DANCE_MOTION.cycle_s,),
    motion_count=1,
    fixed_motion_id=0,
  )
  dq_ref = target_joint_vel(
    env,
    cycle_s=(SINGLE_DANCE_MOTION.cycle_s,),
    motion_count=1,
    fixed_motion_id=0,
  )

  q = robot.data.joint_pos
  dq = robot.data.joint_vel
  pos_error = q_ref - q
  vel_error = dq_ref - dq
  integral_error = torch.clamp(
    integral_error + pos_error * float(env.step_dt),
    min=-INTEGRAL_LIMIT_RAD_S,
    max=INTEGRAL_LIMIT_RAD_S,
  )

  target_delta = (
    KP_TARGET * track_scale * (q_ref - robot.data.default_joint_pos)
    + error_feedback_scale * pos_error
    + KD_TO_POSITION_SECONDS * track_scale * vel_error
    + KI_TO_POSITION_SECONDS * integral_error
  )
  target_delta = target_delta * stability_scale(env)
  target_delta = torch.clamp(target_delta, min=-TARGET_DELTA_LIMIT_RAD, max=TARGET_DELTA_LIMIT_RAD)

  controlled_dim = min(num_actions, target_delta.shape[1])
  action = torch.zeros((target_delta.shape[0], num_actions), device=target_delta.device, dtype=target_delta.dtype)
  action[:, :controlled_dim] = (
    target_delta[:, :controlled_dim]
    / torch.clamp(action_scale[:, :controlled_dim], min=1.0e-6)
  )
  action = torch.clamp(action, min=-1.0, max=1.0)
  action_delta = torch.clamp(
    action - previous_action,
    min=-ACTION_RATE_LIMIT,
    max=ACTION_RATE_LIMIT,
  )
  rate_limited = previous_action + action_delta
  smoothed = (1.0 - ACTION_SMOOTHING) * previous_action + ACTION_SMOOTHING * rate_limited
  return smoothed, integral_error, smoothed.detach()


def play_pid(
  output_dir: str | Path | None = None,
  num_steps: int = DEFAULT_TRACKING_STEPS,
  full_motion: bool = False,
  render_mode: str = "rgb_array",
) -> float:
  if full_motion:
    num_steps = single_motion_frame_count()

  out_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / "outputs"
  env_cfg = _make_pid_env_cfg()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)

  try:
    step_dt = float(env.step_dt)
    wrapped.reset()
    num_actions = infer_num_actions(wrapped, env)
    action_scale = infer_action_scale(env_cfg, num_actions, torch.device(device))
    integral_error = torch.zeros_like(env.scene["robot"].data.joint_pos)
    previous_action = torch.zeros((env.num_envs, num_actions), device=device)
    track_scale, error_feedback_scale = joint_tracking_profile(env, torch.device(device))

    metric_history: list[float] = []
    frames: list[np.ndarray] = []

    for _ in range(num_steps):
      with torch.no_grad():
        action, integral_error, previous_action = joint_target_to_action(
          env,
          action_scale,
          integral_error,
          previous_action,
          track_scale,
          error_feedback_scale,
          num_actions,
        )
      step_result = wrapped.step(action)
      if len(step_result) >= 3:
        done = torch.as_tensor(step_result[2], device=previous_action.device).bool().reshape(-1, 1)
        previous_action = torch.where(done, torch.zeros_like(previous_action), previous_action)
        integral_error = torch.where(done, torch.zeros_like(integral_error), integral_error)

      metric_history.append(
        dance_joint_position_mae(
          env,
          cycle_s=(SINGLE_DANCE_MOTION.cycle_s,),
          motion_count=1,
          fixed_motion_id=0,
        )
      )
      frame = env.render()
      if frame is not None:
        frames.append(frame)

  finally:
    env.close()

  metric_arr = np.asarray(metric_history)
  error = float(np.mean(np.abs(metric_arr)))

  out_dir.mkdir(parents=True, exist_ok=True)
  output_plot = out_dir / "part3_dance_pid_error.png"
  plt.figure()
  plt.plot(metric_arr)
  plt.axhline(0.0, color="r", linestyle="--", label="desired")
  plt.title(f"PID/reference tracking error: {error:.3f} rad")
  plt.xlabel("step")
  plt.ylabel("mean abs joint tracking error [rad]")
  plt.legend()
  plt.savefig(output_plot)
  plt.close()

  output_video = out_dir / "part3_dance_pid_video.mp4"
  if frames:
    fps = int(round(1.0 / step_dt))
    media.write_video(str(output_video), frames, fps=fps)

  print(f"[Part 3 PID baseline] Mean joint tracking error: {error:.3f} rad")
  print(f"[Part 3 PID baseline] Plot: {output_plot}")
  print(f"[Part 3 PID baseline] Video: {output_video}")
  return error


def play_hybrid_pid(
  checkpoint_path: str | Path = Path(__file__).parent / "part3_model_final.pt",
  output_dir: str | Path | None = None,
  num_steps: int = DEFAULT_TRACKING_STEPS,
  render_mode: str = "rgb_array",
) -> float:
  """Use PPO for balance/lower-body control and PID for arm dance tracking."""
  out_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent / "outputs"
  env_cfg = _make_pid_env_cfg()

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  rl_cfg = make_ppo_cfg("hw_part3_dance_single_g1", max_iterations=1)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device=device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True, map_location=device)
  policy = runner.get_inference_policy(device=device)

  try:
    step_dt = float(env.step_dt)
    obs, _ = wrapped.reset()
    num_actions = infer_num_actions(wrapped, env)
    action_scale = infer_action_scale(env_cfg, num_actions, torch.device(device))
    integral_error = torch.zeros_like(env.scene["robot"].data.joint_pos)
    previous_action = torch.zeros((env.num_envs, num_actions), device=device)
    track_scale, error_feedback_scale = joint_tracking_profile(env, torch.device(device))

    arm_mask = torch.zeros((1, num_actions), device=device, dtype=torch.bool)
    for joint_id, name in enumerate(env.scene["robot"].joint_names[:num_actions]):
      if "shoulder" in name or "elbow" in name or "wrist" in name:
        arm_mask[:, joint_id] = True

    metric_history: list[float] = []
    frames: list[np.ndarray] = []

    for _ in range(num_steps):
      with torch.no_grad():
        balance_action = policy(obs)
        pid_action, integral_error, previous_action = joint_target_to_action(
          env,
          action_scale,
          integral_error,
          previous_action,
          track_scale,
          error_feedback_scale,
          num_actions,
        )
        action = torch.where(arm_mask, pid_action, balance_action)
      step_result = wrapped.step(action)
      obs = step_result[0]

      metric_history.append(
        dance_joint_position_mae(
          env,
          cycle_s=(SINGLE_DANCE_MOTION.cycle_s,),
          motion_count=1,
          fixed_motion_id=0,
        )
      )
      frame = env.render()
      if frame is not None:
        frames.append(frame)

  finally:
    env.close()

  metric_arr = np.asarray(metric_history)
  error = float(np.mean(np.abs(metric_arr)))

  out_dir.mkdir(parents=True, exist_ok=True)
  output_plot = out_dir / "part3_dance_hybrid_pid_error.png"
  plt.figure()
  plt.plot(metric_arr)
  plt.axhline(0.0, color="r", linestyle="--", label="desired")
  plt.title(f"Hybrid PID/reference tracking error: {error:.3f} rad")
  plt.xlabel("step")
  plt.ylabel("mean abs joint tracking error [rad]")
  plt.legend()
  plt.savefig(output_plot)
  plt.close()

  output_video = out_dir / "part3_dance_hybrid_pid_video.mp4"
  if frames:
    media.write_video(str(output_video), frames, fps=int(round(1.0 / step_dt)))

  print(f"[Part 3 hybrid PID] Mean joint tracking error: {error:.3f} rad")
  print(f"[Part 3 hybrid PID] Plot: {output_plot}")
  print(f"[Part 3 hybrid PID] Video: {output_video}")
  return error


if __name__ == "__main__":
  play_pid()
