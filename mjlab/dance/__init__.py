"""Single dance trajectory-tracking task helpers for the G1 humanoid.

The homework template in ``scripts`` gives students a mostly complete tracking
framework. This module owns the procedural reference, observation wiring, and
shared evaluation metric so the task stays focused on reward-parameter tuning.

The structure is adapted to the local mjlab stack from the motion-command
tracking layout used by BeyondMimic-style systems: expose phase-conditioned
reference states, track joint targets, and add root-velocity tracking terms.
It does not vendor the upstream Isaac Lab BeyondMimic implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg

from mjlab.tracking_rewards import clipped_square_error, gaussian_reward


@dataclass(frozen=True)
class DanceMotionSpec:
  """Small procedural dance reference used for reward-design homework."""

  name: str
  cycle_s: float
  style_id: int
  motion_path: str | None = None


_HIPHOP_MOTION_PATH = Path(__file__).resolve().parents[2] / "scripts" / "motion" / "g1_hiphop_tracking.npz"
SINGLE_DANCE_MOTION = DanceMotionSpec(
  name="g1_hiphop_tracking",
  cycle_s=7426.0 / 50.0,
  style_id=0,
  motion_path=str(_HIPHOP_MOTION_PATH),
)
DEFAULT_JOINT_POS_STD = 0.35
DEFAULT_JOINT_VEL_STD = 2.0
DEFAULT_ROOT_LIN_VEL_STD = 0.6
DEFAULT_ROOT_ANG_VEL_STD = 1.2
DEFAULT_LINK_POSE_STD = 0.55
DEFAULT_LINK_VEL_STD = 2.5
DEFAULT_TRACKING_STEPS = 600

_ROBOT = SceneEntityCfg("robot")


@dataclass(frozen=True)
class MotionReference:
  fps: float
  joint_names: tuple[str, ...]
  body_names: tuple[str, ...]
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  body_pos_w: np.ndarray
  body_quat_w: np.ndarray
  body_lin_vel_w: np.ndarray
  body_ang_vel_w: np.ndarray
  root_pos_w: np.ndarray
  root_quat_w: np.ndarray
  root_lin_vel_w: np.ndarray
  root_ang_vel_w: np.ndarray

  @property
  def frame_count(self) -> int:
    return int(self.joint_pos.shape[0])

  @property
  def cycle_s(self) -> float:
    return self.frame_count / self.fps


@lru_cache(maxsize=8)
def _load_motion(path: str | None) -> MotionReference | None:
  if path is None:
    return None
  data = np.load(path, allow_pickle=True)
  return MotionReference(
    fps=float(np.asarray(data["fps"]).reshape(-1)[0]),
    joint_names=tuple(str(name) for name in data["joint_names"].tolist()),
    body_names=tuple(str(name) for name in data["body_names"].tolist()),
    joint_pos=np.asarray(data["joint_pos"], dtype=np.float32),
    joint_vel=np.asarray(data["joint_vel"], dtype=np.float32),
    body_pos_w=np.asarray(data["body_pos_w"], dtype=np.float32),
    body_quat_w=np.asarray(data["body_quat_w"], dtype=np.float32),
    body_lin_vel_w=np.asarray(data["body_lin_vel_w"], dtype=np.float32),
    body_ang_vel_w=np.asarray(data["body_ang_vel_w"], dtype=np.float32),
    root_pos_w=np.asarray(data["root_pos_w"], dtype=np.float32),
    root_quat_w=np.asarray(data["root_quat_w"], dtype=np.float32),
    root_lin_vel_w=np.asarray(data["root_lin_vel_w"], dtype=np.float32),
    root_ang_vel_w=np.asarray(data["root_ang_vel_w"], dtype=np.float32),
  )


def _motion_for_id(motion_id: int) -> MotionReference | None:
  if motion_id == 0:
    return _load_motion(SINGLE_DANCE_MOTION.motion_path)
  return None


def _single_motion_reference() -> MotionReference | None:
  return _motion_for_id(0)


def single_motion_frame_count() -> int:
  """Return the frame count of the active single-motion reference."""
  motion = _single_motion_reference()
  if motion is None:
    return DEFAULT_TRACKING_STEPS
  return motion.frame_count


def _as_cycle_tuple(cycle_s: float | Sequence[float]) -> tuple[float, ...]:
  if isinstance(cycle_s, (int, float)):
    return (float(cycle_s),)
  return tuple(float(cycle) for cycle in cycle_s)


def _step_dt(env) -> float:
  return float(getattr(env, "step_dt", 0.02))


def _step_buffer(env, num_envs: int, device: torch.device) -> torch.Tensor:
  for name in ("episode_length_buf", "episode_step_buf", "episode_step"):
    value = getattr(env, name, None)
    if value is not None:
      return value.to(device=device, dtype=torch.float32).reshape(num_envs)
  return torch.zeros(num_envs, device=device)


def _validate_motion_names(asset, motion: MotionReference) -> None:
  if tuple(asset.joint_names) != motion.joint_names:
    raise ValueError("Motion joint_names do not match the G1 asset joint order")
  if tuple(asset.body_names) != motion.body_names:
    raise ValueError("Motion body_names do not match the G1 asset body order")


def _motion_frame_indices(env, motion: MotionReference, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  num_envs = asset.data.joint_pos.shape[0]
  device = asset.data.joint_pos.device
  steps = _step_buffer(env, num_envs, device)
  frame = steps * _step_dt(env) * motion.fps
  if num_envs > 1:
    stagger = torch.arange(num_envs, device=device, dtype=torch.float32) * (motion.frame_count / num_envs)
    frame = frame + stagger
  return torch.remainder(frame.to(dtype=torch.long), motion.frame_count)


def _motion_tensor(
  values: np.ndarray,
  frame_indices: torch.Tensor,
  device: torch.device,
  dtype: torch.dtype,
) -> torch.Tensor:
  indexed = values[frame_indices.detach().cpu().numpy()]
  return torch.as_tensor(indexed, device=device, dtype=dtype)


def _reference_tensor(
  env,
  motion: MotionReference,
  values: np.ndarray,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  frame_indices = _motion_frame_indices(env, motion, asset_cfg)
  return _motion_tensor(values, frame_indices, asset.data.joint_pos.device, asset.data.joint_pos.dtype)


def dance_motion_ids(
  env,
  motion_count: int,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Return the dance id assigned to each parallel environment."""
  asset = env.scene[asset_cfg.name]
  num_envs = asset.data.joint_pos.shape[0]
  device = asset.data.joint_pos.device
  if fixed_motion_id is not None:
    motion_id = int(fixed_motion_id) % motion_count
    return torch.full((num_envs,), motion_id, device=device, dtype=torch.long)
  return torch.arange(num_envs, device=device, dtype=torch.long) % motion_count


def dance_phase(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Return a normalized phase in [0, 1) for each environment."""
  asset = env.scene[asset_cfg.name]
  num_envs = asset.data.joint_pos.shape[0]
  device = asset.data.joint_pos.device
  motion = _single_motion_reference()
  if motion is not None:
    frame_indices = _motion_frame_indices(env, motion, asset_cfg)
    return frame_indices.to(dtype=torch.float32) / float(motion.frame_count)
  steps = _step_buffer(env, num_envs, device)
  cycles = torch.tensor(_as_cycle_tuple(cycle_s), device=device, dtype=torch.float32)
  if cycles.numel() == 1:
    cycle = cycles.expand(num_envs)
  else:
    motion_ids = dance_motion_ids(env, motion_count, fixed_motion_id, asset_cfg)
    cycle = cycles[motion_ids]
  return torch.remainder(steps * _step_dt(env) / cycle, 1.0)


def _dance_offsets(
  default_joint_pos: torch.Tensor,
  phase: torch.Tensor,
  motion_ids: torch.Tensor,
) -> torch.Tensor:
  """Procedural joint offsets for the lightweight dance reference.

  The references are intentionally lightweight: they are deterministic,
  periodic, and defined over whatever joint ordering the installed G1 asset
  exposes. Students optimize tracking rewards, not motion-file parsing.
  """
  num_envs, num_joints = default_joint_pos.shape
  device = default_joint_pos.device
  dtype = default_joint_pos.dtype
  joint_index = torch.arange(num_joints, device=device, dtype=dtype).unsqueeze(0)
  style = motion_ids.to(device=device, dtype=dtype).unsqueeze(1)
  phase = phase.to(device=device, dtype=dtype).unsqueeze(1)

  two_pi = 2.0 * torch.pi
  base_amp = 0.10 + 0.04 * torch.remainder(joint_index, 4.0)
  style_amp = 1.0 + 0.10 * style
  frequency = 1.0 + 0.15 * torch.remainder(joint_index + style, 3.0)
  phase_shift = 0.17 * joint_index + 0.23 * style

  primary = torch.sin(two_pi * (frequency * phase + phase_shift))
  secondary = torch.cos(two_pi * ((0.5 + 0.08 * style) * phase + 0.11 * joint_index))
  offsets = style_amp * base_amp * (primary + 0.35 * secondary)
  return torch.clamp(offsets, min=-0.55, max=0.55)


def target_joint_pos(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reference joint positions for the active dance trajectory."""
  asset = env.scene[asset_cfg.name]
  motion = _single_motion_reference()
  if motion is not None:
    _validate_motion_names(asset, motion)
    return _reference_tensor(env, motion, motion.joint_pos, asset_cfg)
  motion_ids = dance_motion_ids(env, motion_count, fixed_motion_id, asset_cfg)
  phase = dance_phase(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  return asset.data.default_joint_pos + _dance_offsets(
    asset.data.default_joint_pos, phase, motion_ids
  )


def target_joint_vel(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Finite-difference reference joint velocities for the active dance."""
  asset = env.scene[asset_cfg.name]
  motion = _single_motion_reference()
  if motion is not None:
    _validate_motion_names(asset, motion)
    return _reference_tensor(env, motion, motion.joint_vel, asset_cfg)
  motion_ids = dance_motion_ids(env, motion_count, fixed_motion_id, asset_cfg)
  phase = dance_phase(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  dt = _step_dt(env)
  cycles = torch.tensor(_as_cycle_tuple(cycle_s), device=phase.device, dtype=torch.float32)
  if cycles.numel() == 1:
    cycle = cycles.expand_as(phase)
  else:
    cycle = cycles[motion_ids]
  next_phase = torch.remainder(phase + dt / cycle, 1.0)
  current = _dance_offsets(asset.data.default_joint_pos, phase, motion_ids)
  next_pos = _dance_offsets(asset.data.default_joint_pos, next_phase, motion_ids)
  return (next_pos - current) / dt


def target_root_lin_vel(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reference root linear velocity in world frame."""
  asset = env.scene[asset_cfg.name]
  motion = _single_motion_reference()
  if motion is not None:
    _validate_motion_names(asset, motion)
    return _reference_tensor(env, motion, motion.root_lin_vel_w, asset_cfg)
  return torch.zeros_like(asset.data.root_link_lin_vel_b)


def target_root_ang_vel(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reference root angular velocity in world frame."""
  asset = env.scene[asset_cfg.name]
  motion = _single_motion_reference()
  if motion is not None:
    _validate_motion_names(asset, motion)
    return _reference_tensor(env, motion, motion.root_ang_vel_w, asset_cfg)
  return torch.zeros_like(asset.data.root_link_ang_vel_b)


def target_link_pos(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reference link positions in world frame for all robot bodies."""
  asset = env.scene[asset_cfg.name]
  motion = _single_motion_reference()
  if motion is None:
    return asset.data.body_link_pos_w.detach()
  _validate_motion_names(asset, motion)
  return _reference_tensor(env, motion, motion.body_pos_w, asset_cfg)


def target_link_quat(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reference link quaternions in world frame for all robot bodies."""
  asset = env.scene[asset_cfg.name]
  motion = _single_motion_reference()
  if motion is None:
    return asset.data.body_link_quat_w.detach()
  _validate_motion_names(asset, motion)
  return _reference_tensor(env, motion, motion.body_quat_w, asset_cfg)


def target_link_lin_vel(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reference link linear velocities in world frame for all robot bodies."""
  asset = env.scene[asset_cfg.name]
  motion = _single_motion_reference()
  if motion is None:
    return torch.zeros_like(asset.data.body_link_lin_vel_w)
  _validate_motion_names(asset, motion)
  return _reference_tensor(env, motion, motion.body_lin_vel_w, asset_cfg)


def target_link_ang_vel(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reference link angular velocities in world frame for all robot bodies."""
  asset = env.scene[asset_cfg.name]
  motion = _single_motion_reference()
  if motion is None:
    return torch.zeros_like(asset.data.body_link_ang_vel_w)
  _validate_motion_names(asset, motion)
  return _reference_tensor(env, motion, motion.body_ang_vel_w, asset_cfg)


def dance_phase_observation(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  phase = dance_phase(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  angle = 2.0 * torch.pi * phase
  return torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)


def dance_motion_observation(
  env,
  motion_count: int,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  motion_ids = dance_motion_ids(env, motion_count, fixed_motion_id, asset_cfg)
  return torch.nn.functional.one_hot(motion_ids, num_classes=motion_count).float()


def dance_target_observation(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return target_joint_pos(env, cycle_s, motion_count, fixed_motion_id, asset_cfg) - asset.data.default_joint_pos


def dance_target_velocity_observation(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  return target_joint_vel(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)


def dance_root_velocity_error_observation(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  lin_error = target_root_lin_vel(env, cycle_s, motion_count, fixed_motion_id, asset_cfg) - asset.data.root_link_lin_vel_w
  ang_error = target_root_ang_vel(env, cycle_s, motion_count, fixed_motion_id, asset_cfg) - asset.data.root_link_ang_vel_w
  return torch.cat((lin_error, ang_error), dim=1)


def dance_joint_position_mae(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> float:
  asset = env.scene[asset_cfg.name]
  target = target_joint_pos(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  return float(torch.mean(torch.abs(asset.data.joint_pos - target)))


def dance_reference_phase(
  env,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Public wrapper used by richer reference reward implementations."""
  return dance_phase(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)


def track_root_linear_velocity(
  env,
  std: float,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward matching the reference root linear velocity."""
  asset = env.scene[asset_cfg.name]
  reference = target_root_lin_vel(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  error = torch.sum(clipped_square_error(asset.data.root_link_lin_vel_w - reference, limit=1.5), dim=1)
  return gaussian_reward(error, std)


def track_root_angular_velocity(
  env,
  std: float,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward matching the reference root angular velocity."""
  asset = env.scene[asset_cfg.name]
  reference = target_root_ang_vel(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  error = torch.sum(clipped_square_error(asset.data.root_link_ang_vel_w - reference, limit=3.0), dim=1)
  return gaussian_reward(error, std)


def track_link_pose(
  env,
  std: float,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward matching all link positions and orientations from the motion file."""
  asset = env.scene[asset_cfg.name]
  reference_pos = target_link_pos(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  reference_quat = target_link_quat(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)

  actual_rel_pos = asset.data.body_link_pos_w - asset.data.body_link_pos_w[:, :1]
  reference_rel_pos = reference_pos - reference_pos[:, :1]
  pos_error = torch.mean(
    torch.sum(clipped_square_error(actual_rel_pos - reference_rel_pos, limit=1.0), dim=-1),
    dim=1,
  )

  actual_quat = torch.nn.functional.normalize(asset.data.body_link_quat_w, dim=-1)
  reference_quat = torch.nn.functional.normalize(reference_quat, dim=-1)
  quat_dot = torch.sum(actual_quat * reference_quat, dim=-1).abs().clamp(max=1.0)
  quat_error = torch.mean(1.0 - quat_dot * quat_dot, dim=1)

  return gaussian_reward(pos_error + 0.5 * quat_error, std)


def track_link_velocity(
  env,
  std: float,
  cycle_s: float | Sequence[float],
  motion_count: int = 1,
  fixed_motion_id: int | None = None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward matching all link linear and angular velocities from the motion file."""
  asset = env.scene[asset_cfg.name]
  reference_lin = target_link_lin_vel(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  reference_ang = target_link_ang_vel(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  lin_error = torch.mean(
    torch.sum(clipped_square_error(asset.data.body_link_lin_vel_w - reference_lin, limit=5.0), dim=-1),
    dim=1,
  )
  ang_error = torch.mean(
    torch.sum(clipped_square_error(asset.data.body_link_ang_vel_w - reference_ang, limit=8.0), dim=-1),
    dim=1,
  )
  error = lin_error + 0.25 * ang_error
  return gaussian_reward(error, std)


def _freeze_velocity_command(cfg: ManagerBasedRlEnvCfg) -> None:
  twist = cfg.commands.get("twist") if hasattr(cfg, "commands") else None
  if twist is None:
    return
  if hasattr(twist, "ranges"):
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = (0.0, 0.0)
    twist.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(twist.ranges, "heading"):
      twist.ranges.heading = None
  if hasattr(twist, "heading_command"):
    twist.heading_command = False


def _add_dance_observations(
  cfg: ManagerBasedRlEnvCfg,
  cycle_s: float | Sequence[float],
  motion_count: int,
  fixed_motion_id: int | None,
) -> None:
  common_params = {
    "cycle_s": cycle_s,
    "motion_count": motion_count,
    "fixed_motion_id": fixed_motion_id,
  }
  for group in cfg.observations.values():
    group.terms["dance_phase"] = ObservationTermCfg(
      func=dance_phase_observation,
      params=dict(common_params),
    )
    group.terms["dance_target_joint_pos"] = ObservationTermCfg(
      func=dance_target_observation,
      params=dict(common_params),
    )
    group.terms["dance_target_joint_vel"] = ObservationTermCfg(
      func=dance_target_velocity_observation,
      params=dict(common_params),
    )
    group.terms["dance_root_velocity_error"] = ObservationTermCfg(
      func=dance_root_velocity_error_observation,
      params=dict(common_params),
    )
    if motion_count > 1:
      group.terms["dance_motion_id"] = ObservationTermCfg(
        func=dance_motion_observation,
        params={"motion_count": motion_count, "fixed_motion_id": fixed_motion_id},
      )


def make_dance_env_cfg(
  track_joint_pos_fn: Callable[..., torch.Tensor],
  track_joint_vel_fn: Callable[..., torch.Tensor],
  motions: Sequence[DanceMotionSpec],
  fixed_motion_id: int | None = None,
  joint_pos_std: float = DEFAULT_JOINT_POS_STD,
  joint_vel_std: float = DEFAULT_JOINT_VEL_STD,
  root_lin_vel_std: float = DEFAULT_ROOT_LIN_VEL_STD,
  root_ang_vel_std: float = DEFAULT_ROOT_ANG_VEL_STD,
  link_pose_std: float = DEFAULT_LINK_POSE_STD,
  link_vel_std: float = DEFAULT_LINK_VEL_STD,
  joint_pos_weight: float = 3.0,
  joint_vel_weight: float = 0.5,
  root_lin_vel_weight: float = 0.35,
  root_ang_vel_weight: float = 0.25,
  link_pose_weight: float = 1.0,
  link_vel_weight: float = 0.35,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a G1 dance-tracking env from the flat-walk base config."""
  if not motions:
    raise ValueError("motions must contain at least one DanceMotionSpec")

  cfg = unitree_g1_flat_env_cfg(play=play)
  _freeze_velocity_command(cfg)

  cycle_s = tuple(motion.cycle_s for motion in motions)
  motion_count = len(motions)
  _add_dance_observations(cfg, cycle_s, motion_count, fixed_motion_id)

  reward_params = {
    "cycle_s": cycle_s,
    "motion_count": motion_count,
    "fixed_motion_id": fixed_motion_id,
  }
  cfg.rewards.pop("track_linear_velocity", None)
  cfg.rewards.pop("track_angular_velocity", None)
  cfg.rewards["track_dance_joint_position"] = RewardTermCfg(
    func=track_joint_pos_fn,
    weight=joint_pos_weight,
    params={"std": joint_pos_std, **reward_params},
  )
  cfg.rewards["track_dance_joint_velocity"] = RewardTermCfg(
    func=track_joint_vel_fn,
    weight=joint_vel_weight,
    params={"std": joint_vel_std, **reward_params},
  )
  cfg.rewards["track_dance_root_linear_velocity"] = RewardTermCfg(
    func=track_root_linear_velocity,
    weight=root_lin_vel_weight,
    params={"std": root_lin_vel_std, **reward_params},
  )
  cfg.rewards["track_dance_root_angular_velocity"] = RewardTermCfg(
    func=track_root_angular_velocity,
    weight=root_ang_vel_weight,
    params={"std": root_ang_vel_std, **reward_params},
  )
  cfg.rewards["track_dance_link_pose"] = RewardTermCfg(
    func=track_link_pose,
    weight=link_pose_weight,
    params={"std": link_pose_std, **reward_params},
  )
  cfg.rewards["track_dance_link_velocity"] = RewardTermCfg(
    func=track_link_velocity,
    weight=link_vel_weight,
    params={"std": link_vel_std, **reward_params},
  )
  cfg.episode_length_s = 8.0 if play else 6.0
  return cfg


def make_single_dance_env_cfg(
  track_joint_pos_fn: Callable[..., torch.Tensor],
  track_joint_vel_fn: Callable[..., torch.Tensor],
  joint_pos_std: float = DEFAULT_JOINT_POS_STD,
  joint_vel_std: float = DEFAULT_JOINT_VEL_STD,
  root_lin_vel_std: float = DEFAULT_ROOT_LIN_VEL_STD,
  root_ang_vel_std: float = DEFAULT_ROOT_ANG_VEL_STD,
  link_pose_std: float = DEFAULT_LINK_POSE_STD,
  link_vel_std: float = DEFAULT_LINK_VEL_STD,
  joint_pos_weight: float = 3.0,
  joint_vel_weight: float = 0.5,
  root_lin_vel_weight: float = 0.35,
  root_ang_vel_weight: float = 0.25,
  link_pose_weight: float = 1.0,
  link_vel_weight: float = 0.35,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  return make_dance_env_cfg(
    track_joint_pos_fn,
    track_joint_vel_fn,
    motions=(SINGLE_DANCE_MOTION,),
    fixed_motion_id=0,
    joint_pos_std=joint_pos_std,
    joint_vel_std=joint_vel_std,
    root_lin_vel_std=root_lin_vel_std,
    root_ang_vel_std=root_ang_vel_std,
    link_pose_std=link_pose_std,
    link_vel_std=link_vel_std,
    joint_pos_weight=joint_pos_weight,
    joint_vel_weight=joint_vel_weight,
    root_lin_vel_weight=root_lin_vel_weight,
    root_ang_vel_weight=root_ang_vel_weight,
    link_pose_weight=link_pose_weight,
    link_vel_weight=link_vel_weight,
    play=play,
  )
