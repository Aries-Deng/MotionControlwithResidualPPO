"""Shared evaluation helper: roll out a trained policy, save metric plot + video."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper


def evaluate(
  env_cfg: ManagerBasedRlEnvCfg,
  rl_cfg: RslRlOnPolicyRunnerCfg,
  checkpoint_path: str | Path,
  metric_fn: Callable[[ManagerBasedRlEnv], float],
  metric_label: str,
  desired_value: float,
  num_steps: int,
  output_plot: str | Path,
  output_video: str | Path | None,
  render_mode: str = "rgb_array",
  device: str | None = None,
) -> float:
  """Run a deterministic rollout and dump a metric plot + an mp4.

  Args:
    env_cfg: env config (typically built with play=True).
    rl_cfg: matching RL runner config.
    checkpoint_path: .pt file produced by training.
    metric_fn: maps the env to a scalar measured each step.
    metric_label: y-axis label.
    desired_value: dashed reference line drawn on the plot.
    num_steps: number of env steps to roll out (single env).
    output_plot: where to save the metric plot.
    output_video: where to save the rollout video. Ignored if no frames are returned.
    render_mode: "rgb_array" for mp4 export or "human" for an interactive viewer.

  Returns:
    Mean absolute error of metric vs. desired_value over the rollout.
  """
  device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  # Force single env for clean metrics + rendering.
  env_cfg.scene.num_envs = 1

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=rl_cfg.clip_actions)

  runner = MjlabOnPolicyRunner(wrapped, asdict(rl_cfg), device=device)
  runner.load(str(checkpoint_path), load_cfg={"actor": True}, strict=True,
              map_location=device)
  policy = runner.get_inference_policy(device=device)

  obs, _ = wrapped.reset()
  metric_history: list[float] = []
  frames: list[np.ndarray] = []

  for _ in range(num_steps):
    with torch.no_grad():
      action = policy(obs)
    step_result = wrapped.step(action)
    obs = step_result[0]
    metric_history.append(float(metric_fn(env)))
    frame = env.render()
    if frame is not None:
      frames.append(frame)

  metric_arr = np.asarray(metric_history)
  error = float(np.mean(np.abs(metric_arr - desired_value)))

  Path(output_plot).parent.mkdir(parents=True, exist_ok=True)
  plt.figure()
  plt.plot(metric_arr)
  plt.axhline(desired_value, color="r", linestyle="--", label="desired")
  plt.title(f"{metric_label} error: {error:.3f}")
  plt.xlabel("step")
  plt.ylabel(metric_label)
  plt.legend()
  plt.savefig(output_plot)
  plt.close()

  if frames and output_video is not None:
    Path(output_video).parent.mkdir(parents=True, exist_ok=True)
    fps = int(round(1.0 / env.step_dt))
    media.write_video(str(output_video), frames, fps=fps)

  env.close()
  return error
