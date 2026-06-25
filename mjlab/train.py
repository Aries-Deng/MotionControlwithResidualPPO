"""Shared training helper: run rsl_rl PPO on a given env_cfg + rl_cfg."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.utils.os import dump_yaml
from mjlab.utils.torch import configure_torch_backends


def train(
  env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
  num_envs: int = 4096,
  device: str | None = None,
  log_root: str | Path = "logs",
) -> Path:
  """Train a PPO policy on `env_cfg` and return the directory containing checkpoints."""
  configure_torch_backends()
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg.scene.num_envs = num_envs

  log_dir = (
    Path(log_root) / "rsl_rl" / rl_cfg.experiment_name
    / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  )
  log_dir.mkdir(parents=True, exist_ok=True)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), str(log_dir), device)
  runner.add_git_repo_to_log(__file__)

  dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
  dump_yaml(log_dir / "params" / "agent.yaml", asdict(rl_cfg))

  runner.learn(
    num_learning_iterations=rl_cfg.max_iterations,
    init_at_random_ep_len=True,
  )
  env.close()
  return log_dir


def latest_checkpoint(log_dir: Path) -> Path:
  """Return the most recent .pt checkpoint inside log_dir."""
  ckpts = sorted(log_dir.glob("model_*.pt"))
  if not ckpts:
    raise FileNotFoundError(f"No model_*.pt checkpoint under {log_dir}")
  return ckpts[-1]
