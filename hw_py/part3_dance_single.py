"""Part 3 - Single dance trajectory tracking (Unitree G1 humanoid).

The tracking framework follows a local BeyondMimic-style layout: the policy
observes phase-conditioned reference states and learns from multi-term motion
tracking rewards. Tune the small set of reward parameters below, train a few
times, and use the tracking-error plot/video to choose a stable setting.

Run this file to train a PPO policy and produce:
  - hw_py/part3_dance_single_error.png
  - hw_py/part3_dance_single_video.mp4
"""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from hw_mjlab.dance import (
  DEFAULT_TRACKING_STEPS,
  SINGLE_DANCE_MOTION,
  dance_reference_phase,
  dance_joint_position_mae,
  make_single_dance_env_cfg,
  single_motion_frame_count,
  target_joint_pos,
  target_joint_vel,
)
from hw_mjlab.rl_cfg import make_ppo_cfg
from hw_mjlab.tracking_rewards import (
  clipped_square_error,
  gaussian_reward,
  joint_tracking_weights,
  phase_tracking_gain,
  weighted_mean,
)


_ROBOT = SceneEntityCfg("robot")


# =============================================================================
# TODO - tune only these four interpretable parameters.
# =============================================================================
# Recommended ranges:
#   JOINT_POS_STD:    0.20 to 0.70
#   LINK_POSE_STD:    0.30 to 1.00
#   JOINT_POS_WEIGHT: 1.5  to 5.0
#   LINK_POSE_WEIGHT: 0.4  to 1.8
#
# Suggested workflow: run the defaults once, inspect the tracking plot/video,
# then change 1-2 values at a time. The defaults below are intentionally
# serviceable but not optimal, so students should be able to improve them with
# a few short runs.
# =============================================================================

JOINT_POS_STD = 0.35
LINK_POSE_STD = 0.55
JOINT_POS_WEIGHT = 3.0
LINK_POSE_WEIGHT = 1.0


# Keep the remaining parameters fixed.
JOINT_VEL_STD = 2.0
ROOT_LIN_VEL_STD = 0.6
ROOT_ANG_VEL_STD = 1.2
LINK_VEL_STD = 2.5
JOINT_VEL_WEIGHT = 0.5
ROOT_LIN_VEL_WEIGHT = 0.35
ROOT_ANG_VEL_WEIGHT = 0.25
LINK_VEL_WEIGHT = 0.35
JOINT_POS_ERROR_CLIP = 0.8
JOINT_VEL_ERROR_CLIP = 8.0


def track_joint_position(
  env,
  std: float,
  cycle_s: tuple[float, ...],
  motion_count: int,
  fixed_motion_id: int | None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward tracking the reference joint positions for the single dance."""
  asset: Entity = env.scene[asset_cfg.name]
  reference = target_joint_pos(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  phase = dance_reference_phase(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  weights = joint_tracking_weights(reference)
  position_error = clipped_square_error(
    asset.data.joint_pos - reference,
    limit=JOINT_POS_ERROR_CLIP,
  )
  shaped_error = weighted_mean(position_error, weights) * phase_tracking_gain(phase)
  return gaussian_reward(shaped_error, std)


def track_joint_velocity(
  env,
  std: float,
  cycle_s: tuple[float, ...],
  motion_count: int,
  fixed_motion_id: int | None,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward tracking the reference joint velocities for the single dance."""
  asset: Entity = env.scene[asset_cfg.name]
  reference = target_joint_vel(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  reference_pos = target_joint_pos(env, cycle_s, motion_count, fixed_motion_id, asset_cfg)
  weights = joint_tracking_weights(reference_pos)
  velocity_error = clipped_square_error(
    asset.data.joint_vel - reference,
    limit=JOINT_VEL_ERROR_CLIP,
  )
  return gaussian_reward(weighted_mean(velocity_error, weights), std)


def _make_tracking_env_cfg(play: bool = False):
  return make_single_dance_env_cfg(
    track_joint_position,
    track_joint_velocity,
    joint_pos_std=JOINT_POS_STD,
    joint_vel_std=JOINT_VEL_STD,
    root_lin_vel_std=ROOT_LIN_VEL_STD,
    root_ang_vel_std=ROOT_ANG_VEL_STD,
    link_pose_std=LINK_POSE_STD,
    link_vel_std=LINK_VEL_STD,
    joint_pos_weight=JOINT_POS_WEIGHT,
    joint_vel_weight=JOINT_VEL_WEIGHT,
    root_lin_vel_weight=ROOT_LIN_VEL_WEIGHT,
    root_ang_vel_weight=ROOT_ANG_VEL_WEIGHT,
    link_pose_weight=LINK_POSE_WEIGHT,
    link_vel_weight=LINK_VEL_WEIGHT,
    play=play,
  )


def play(
  checkpoint_path: str | Path,
  output_dir: str | Path | None = None,
  num_steps: int = DEFAULT_TRACKING_STEPS,
  viewer: bool = False,
  full_motion: bool = False,
) -> float:
  """Run single-trajectory play/evaluation from an existing checkpoint."""
  from hw_mjlab.evaluate import evaluate

  if viewer:
    raise NotImplementedError(
      "viewer=True is not supported by the current mjlab build. "
      "ManagerBasedRlEnv only exposes render modes None and 'rgb_array', "
      "so use the saved mp4 output instead."
    )

  if full_motion:
    num_steps = single_motion_frame_count()

  out_dir = Path(output_dir) if output_dir is not None else Path(__file__).parent
  rl_cfg = make_ppo_cfg("hw_part3_dance_single_g1", max_iterations=1)
  error = evaluate(
    env_cfg=_make_tracking_env_cfg(play=True),
    rl_cfg=rl_cfg,
    checkpoint_path=checkpoint_path,
    metric_fn=lambda env: dance_joint_position_mae(
      env,
      cycle_s=(SINGLE_DANCE_MOTION.cycle_s,),
      motion_count=1,
      fixed_motion_id=0,
    ),
    metric_label="mean abs joint tracking error [rad]",
    desired_value=0.0,
    num_steps=num_steps,
    output_plot=out_dir / "part3_dance_single_error.png",
    output_video=out_dir / "part3_dance_single_video.mp4",
    render_mode="rgb_array",
  )
  print(f"[Part 3 play] Single-dance mean joint tracking error: {error:.3f} rad")
  return error


def main(num_envs: int = 4096, max_iterations: int = 2500) -> None:
  from hw_mjlab.train import latest_checkpoint, train

  rl_cfg = make_ppo_cfg("hw_part3_dance_single_g1", max_iterations=max_iterations)
  log_dir = train(_make_tracking_env_cfg(), rl_cfg, num_envs=num_envs)
  ckpt = latest_checkpoint(log_dir)
  play(ckpt)


if __name__ == "__main__":
  main()
