"""Reusable reward-building blocks for walking and dance tracking."""

from __future__ import annotations

import torch


def clipped_square_error(error: torch.Tensor, limit: float) -> torch.Tensor:
  """Square error with an outlier cap so bad early rollouts do not dominate."""
  return torch.square(torch.clamp(error, min=-limit, max=limit))


def gaussian_reward(error: torch.Tensor, std: float) -> torch.Tensor:
  return torch.exp(-error / (std * std))


def joint_tracking_weights(reference: torch.Tensor) -> torch.Tensor:
  """Heuristic joint weights for the procedural G1 dance references.

  The procedural reference is ordered by the installed G1 asset. We avoid
  relying on joint names in the student solution, but still weight high-motion
  reference joints more than near-static joints.
  """
  amplitude = torch.mean(torch.abs(reference - torch.mean(reference, dim=1, keepdim=True)), dim=0)
  weights = 1.0 + 2.0 * amplitude / torch.clamp(torch.mean(amplitude), min=1.0e-4)
  return torch.clamp(weights, min=0.75, max=4.0)


def weighted_mean(error: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
  weights = weights.to(device=error.device, dtype=error.dtype)
  return torch.sum(error * weights.unsqueeze(0), dim=1) / torch.sum(weights)


def phase_tracking_gain(phase: torch.Tensor) -> torch.Tensor:
  """Slightly emphasize the expressive middle of each dance beat."""
  return 0.75 + 0.25 * torch.sin(2.0 * torch.pi * phase).abs()
