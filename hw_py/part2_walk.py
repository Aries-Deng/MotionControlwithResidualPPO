"""Part 2 - Pure walking velocity tracking (Unitree G1 humanoid).

The G1 must follow joystick-style velocity commands on flat ground. The base
mjlab G1 velocity task already provides the main proprioceptive observations
and regularization rewards. You will design a small observation augmentation
and replace only the two command-tracking rewards.

Run this file to train a PPO policy and produce:
  - hw_py/part2_walk_LinVel_error.png
  - hw_py/part2_walk_video.mp4
"""

from __future__ import annotations

from pathlib import Path

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from hw_mjlab.rl_cfg import make_ppo_cfg
from hw_mjlab.walk import make_walk_env_cfg
from hw_mjlab.tracking_rewards import clipped_square_error, gaussian_reward


DESIRED_LIN_VEL_X = 1.0
_ROBOT = SceneEntityCfg("robot")


# Inherited actor observations from the base task:
#   base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel,
#   actions, command
# Inherited critic-only observations:
#   foot_height, foot_air_time, foot_contact, foot_contact_forces
# Inherited reward/regularization terms kept by make_walk_env_cfg:
#   upright, pose, body_ang_vel, angular_momentum, dof_pos_limits,
#   action_rate_l2, foot_clearance, foot_swing_height, foot_slip,
#   soft_landing, self_collisions
# You only add two compact observation terms and replace the two tracking
# rewards below. Do not remove the inherited terms; they are what make the
# resulting policy stable enough to train into a usable walk.


# =============================================================================
# TODO A - observation design for direct velocity walking.
# =============================================================================
# These two terms are appended to the base policy observation. Keep them compact:
# expose command-tracking and stability information that helps the policy learn
# velocity walking directly, without adding a motion imitation objective.
# =============================================================================


def observe_velocity_error(
  env,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Return compact command-tracking features, e.g. commanded minus actual twist."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual_lin = asset.data.root_link_lin_vel_b
  actual_ang = asset.data.root_link_ang_vel_b
  # TODO: return Tensor[B, k]. A good minimal choice is
  # [cmd_vx - vx, cmd_vy - vy, cmd_yaw - yaw_rate].
  lin_error = command[:, :2] - actual_lin[:, :2]
  yaw_error = command[:, 2:3] - actual_ang[:, 2:3]
  return torch.cat((lin_error, yaw_error), dim=1)


def observe_stability_context(
  env,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Return compact balance features useful for velocity walking."""
  asset: Entity = env.scene[asset_cfg.name]
  # TODO: return Tensor[B, k]. Useful signals include projected gravity,
  # body angular velocity, or vertical body velocity.
  projected_gravity = asset.data.projected_gravity_b
  roll_pitch_rate = asset.data.root_link_ang_vel_b[:, :2]
  vertical_velocity = asset.data.root_link_lin_vel_b[:, 2:3]
  return torch.cat((projected_gravity, roll_pitch_rate, vertical_velocity), dim=1)


# =============================================================================
# TODO B - implement the two walking command-tracking rewards.
# =============================================================================
# Useful data:
#   robot.data.root_link_lin_vel_b   [B, 3] base linear velocity in body frame
#   robot.data.root_link_ang_vel_b   [B, 3] base angular velocity in body frame
#   command[:, :2]                   target xy linear velocity
#   command[:, 2]                    target yaw rate
#
# Both rewards should be positive. A common starting shape is:
#   exp(-squared_error / std**2)
# Good solutions often weight forward/lateral/yaw errors differently, clip large
# early-training errors, and lightly discourage vertical drift or roll/pitch
# angular velocity.
# =============================================================================


def track_linear_velocity(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward tracking the commanded xy linear velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_lin_vel_b
  # TODO: compute a per-env tracking reward for actual[:, :2] vs command[:, :2].
  # You may use weighted xy errors and optional vertical-drift shaping.
  xy_error = actual[:, :2] - command[:, :2]
  clipped_xy_error = clipped_square_error(xy_error, limit=1.5)
  weighted_xy_error = clipped_xy_error[:, 0] + 1.5 * clipped_xy_error[:, 1]

  vertical_error = clipped_square_error(actual[:, 2], limit=1.0)
  shaped_error = weighted_xy_error + 0.15 * vertical_error
  return gaussian_reward(shaped_error, std)


def track_angular_velocity(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _ROBOT,
) -> torch.Tensor:
  """Reward tracking the commanded yaw angular velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual = asset.data.root_link_ang_vel_b
  # TODO: compute a per-env tracking reward for actual[:, 2] vs command[:, 2].
  # You may lightly penalize roll/pitch angular velocity for smoother walking.
  yaw_error = actual[:, 2] - command[:, 2]
  clipped_yaw_error = clipped_square_error(yaw_error, limit=3.0)

  roll_pitch_error = torch.sum(clipped_square_error(actual[:, :2], limit=3.0), dim=1)
  shaped_error = clipped_yaw_error + 0.05 * roll_pitch_error
  return gaussian_reward(shaped_error, std)


def _set_fixed_forward_command(play_cfg) -> None:
  twist = play_cfg.commands["twist"]
  twist.ranges.lin_vel_x = (DESIRED_LIN_VEL_X, DESIRED_LIN_VEL_X)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  if hasattr(twist.ranges, "heading"):
    twist.ranges.heading = None
  twist.heading_command = False


def main(num_envs: int = 4096, max_iterations: int = 2500) -> None:
  from hw_mjlab.evaluate import evaluate
  from hw_mjlab.train import latest_checkpoint, train

  env_cfg = make_walk_env_cfg(
    track_linear_velocity,
    track_angular_velocity,
    observe_velocity_error,
    observe_stability_context,
  )
  rl_cfg = make_ppo_cfg("hw_part2_walk_g1", max_iterations=max_iterations)
  log_dir = train(env_cfg, rl_cfg, num_envs=num_envs)
  ckpt = latest_checkpoint(log_dir)

  play_cfg = make_walk_env_cfg(
    track_linear_velocity,
    track_angular_velocity,
    observe_velocity_error,
    observe_stability_context,
    play=True,
  )
  _set_fixed_forward_command(play_cfg)

  out_dir = Path(__file__).parent
  error = evaluate(
    env_cfg=play_cfg,
    rl_cfg=rl_cfg,
    checkpoint_path=ckpt,
    metric_fn=lambda env: float(env.scene["robot"].data.root_link_lin_vel_b[0, 0]),
    metric_label="forward linear velocity [m/s]",
    desired_value=DESIRED_LIN_VEL_X,
    num_steps=500,
    output_plot=out_dir / "part2_walk_LinVel_error.png",
    output_video=out_dir / "part2_walk_video.mp4",
  )
  print(f"[Part 2] Mean forward-velocity error: {error:.3f} m/s")


if __name__ == "__main__":
  main()
