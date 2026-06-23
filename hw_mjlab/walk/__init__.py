"""Flat-ground G1 base config for the pure velocity-walking task."""

from __future__ import annotations

from typing import Callable

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg


BASE_ACTOR_OBSERVATIONS = (
  "base_lin_vel",
  "base_ang_vel",
  "projected_gravity",
  "joint_pos",
  "joint_vel",
  "actions",
  "command",
)
BASE_CRITIC_EXTRA_OBSERVATIONS = (
  "foot_height",
  "foot_air_time",
  "foot_contact",
  "foot_contact_forces",
)
BASE_REGULARIZATION_REWARDS = (
  "upright",
  "pose",
  "body_ang_vel",
  "angular_momentum",
  "dof_pos_limits",
  "action_rate_l2",
  "foot_clearance",
  "foot_swing_height",
  "foot_slip",
  "soft_landing",
  "self_collisions",
)


def _require_terms(container: dict, names: tuple[str, ...], label: str) -> None:
  missing = [name for name in names if name not in container]
  if missing:
    raise RuntimeError(f"Base walk config is missing {label}: {missing}")


def _validate_base_walk_cfg(cfg: ManagerBasedRlEnvCfg) -> None:
  _require_terms(cfg.observations["actor"].terms, BASE_ACTOR_OBSERVATIONS, "actor observations")
  _require_terms(cfg.observations["critic"].terms, BASE_ACTOR_OBSERVATIONS, "critic observations")
  _require_terms(
    cfg.observations["critic"].terms,
    BASE_CRITIC_EXTRA_OBSERVATIONS,
    "critic-only observations",
  )
  _require_terms(cfg.rewards, BASE_REGULARIZATION_REWARDS, "regularization rewards")


def make_walk_env_cfg(
  track_lin_vel_fn: Callable[..., torch.Tensor],
  track_ang_vel_fn: Callable[..., torch.Tensor],
  observe_velocity_error_fn: Callable[..., torch.Tensor] | None = None,
  observe_stability_context_fn: Callable[..., torch.Tensor] | None = None,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build a G1 flat-ground env config for direct velocity walking.

  The final policy is not trained with only the two tracking rewards. The base
  config already provides actor observations such as base velocity, gravity,
  joint state, previous actions, and command; the critic additionally receives
  foot-contact context. It also keeps the official upright, pose, angular-rate,
  joint-limit, action-rate, foot-slip, clearance, landing, and self-collision
  reward terms. This helper only replaces the two command-tracking rewards and
  appends the two student-designed observation terms.

  Args:
    track_lin_vel_fn: student-implemented linear-velocity tracking reward.
        Signature: (env, command_name: str, std: float, asset_cfg) -> Tensor[B].
    track_ang_vel_fn: student-implemented angular-velocity tracking reward.
        Signature: (env, command_name: str, std: float, asset_cfg) -> Tensor[B].
    observe_velocity_error_fn: optional student observation term exposing
      velocity-command error or other compact command-tracking state.
    observe_stability_context_fn: optional student observation term exposing
      balance-related context such as gravity or angular velocity.
    play: if True, builds the play-mode config (no DR, infinite episode).
  """
  cfg = unitree_g1_flat_env_cfg(play=play)
  _validate_base_walk_cfg(cfg)

  # Swap in the student's reward functions, keep weights/params intact.
  rewards = cfg.rewards
  rewards["track_linear_velocity"] = RewardTermCfg(
    func=track_lin_vel_fn,
    weight=rewards["track_linear_velocity"].weight,
    params=dict(rewards["track_linear_velocity"].params),
  )
  rewards["track_angular_velocity"] = RewardTermCfg(
    func=track_ang_vel_fn,
    weight=rewards["track_angular_velocity"].weight,
    params=dict(rewards["track_angular_velocity"].params),
  )

  if observe_velocity_error_fn is not None:
    for group in cfg.observations.values():
      group.terms["student_velocity_error"] = ObservationTermCfg(
        func=observe_velocity_error_fn,
        params={"command_name": "twist"},
      )
  if observe_stability_context_fn is not None:
    for group in cfg.observations.values():
      group.terms["student_stability_context"] = ObservationTermCfg(
        func=observe_stability_context_fn,
      )
  return cfg
