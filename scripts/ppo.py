"""
Part 1 guidance: how to fill the NumPy PPO TODOs.

Suggested workflow for each function:
1. Convert inputs to np.asarray(...) if you want predictable shapes.
2. Work out the target formula on paper first.
3. Pre-allocate an output array when the result is a sequence.
4. Pay attention to whether the loop should run forward or backward.
5. Return a scalar for losses and a vector for advantages.

Minimal example scaffold (this is only a shape-handling example, not the
solution to any TODO below):

    rewards = np.asarray(rewards, dtype=float)
    values = np.asarray(values, dtype=float)
    T = len(rewards)
    advantages = np.zeros(T, dtype=float)

    # fill advantages[t] inside a loop
    return advantages

Useful shape reminder:
- rewards has shape (T,)
- values has shape (T + 1,)
- most advantage outputs have shape (T,)

Toy example (3 timesteps):

    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.5, 0.4, 0.3, 0.0])
    gamma = 0.9

    # Monte Carlo return is easiest to compute backward:
    # G_2 = 3.0
    # G_1 = 2.0 + 0.9 * 3.0 = 4.7
    # G_0 = 1.0 + 0.9 * 4.7 = 5.23
    # so MC advantages would be:
    # [5.23 - 0.5, 4.7 - 0.4, 3.0 - 0.3]

    # TD residual at t = 1 is local:
    # delta_1 = r_1 + gamma * V(s_2) - V(s_1)
    #         = 2.0 + 0.9 * 0.3 - 0.4
    #         = 1.87

    # Value loss toy example:
    # values  = np.array([1.0, 2.0])
    # returns = np.array([1.5, 1.0])
    # mse = mean([(1.0 - 1.5)^2, (2.0 - 1.0)^2]) = 0.625
"""

import numpy as np

# Monte Carlo Advantage
def monte_carlo_advantage(rewards: np.ndarray, values: np.ndarray, gamma: float):
    """
    Monte Carlo advantage estimation.

    Args:
        rewards (np.ndarray): sequence of rewards with shape (T,).
        values (np.ndarray): sequence of estimated state values with shape (T+1,).
        gamma (float): discount factor.

    Returns:
        advantages: (np.array) Gt - V(s)
    """
    # Hint:
    # 1. Let G_t be the discounted return starting from timestep t.
    # 2. Compute returns backward so you can reuse the next-step result.
    # 3. After getting G_t, subtract values[t] to form the advantage.
    # 4. Output shape should be (T,), not (T + 1,).
    ...

def td_residual_advantage(rewards: np.ndarray, values: np.ndarray, gamma: float):
    """
    TD(0) residual advantage estimation (one-step TD error).

    Args:
        rewards: list or np.array of rewards with shape (T,).
        values: list or np.array of values  with shape (T+1,).
        gamma: discount factor.

    Returns:
        advantages: (np.array) δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    """
    # Hint:
    # 1. This is the one-step TD error, so each timestep is independent.
    # 2. You can fill advantages[t] directly from rewards[t], values[t],
    #    and values[t + 1].
    # 3. A forward loop is enough here.
    ...


def generalized_advantage_estimation(rewards, values, gamma, lam):
    """
    Generalized Advantage Estimation (GAE).

    Args:
        rewards: list or np.array of rewards.
        values: list or np.array of values (length = len(rewards) + 1).
        gamma: discount factor.
        lam: GAE lambda parameter (between 0 and 1).
               λ=0: reduces to TD(0) (high bias, low variance).
               λ=1: reduces to Monte Carlo (low bias, high variance).

    Returns:
        advantages: (np.array) GAE advantages
    """
    # Hint:
    # 1. First compute delta_t = r_t + gamma * V(s_{t+1}) - V(s_t).
    # 2. Then accumulate GAE backward with the recursive form that uses
    #    gamma * lam on the next accumulated term.
    # 3. When lam = 0, your result should match TD residual advantage.
    # 4. When lam is close to 1, it should behave more like Monte Carlo.
    ...


def compute_policy_loss(ratio, adv, dist_entropy, epsilon, entropy_weight):
    """
    Compute the policy (actor) loss for PPO using NumPy.

    Args:
        ratio (np.ndarray): Probability ratios between new and old policies.
        adv (np.ndarray): Advantage estimates.
        dist_entropy (float): Precomputed mean entropy of the new policy distribution.
        epsilon (float): PPO clip range.
        entropy_weight (float): Entropy bonus weight.

    Returns:
        float: The computed policy loss (scalar).
    """
    # Hint:
    # 1. Form the unclipped surrogate with ratio * adv.
    # 2. Clip ratio into [1 - epsilon, 1 + epsilon].
    # 3. Build the clipped surrogate and take the elementwise minimum.
    # 4. PPO usually optimizes the negative mean surrogate, then adds the
    #    entropy bonus with the correct sign.
    ...


def compute_value_loss(values, returns):
    """
    Compute the value loss for PPO using NumPy. The loss should be Mean Squared Error (MSE) between predicted values and target returns.

    Args:
        values (np.ndarray): Predicted state values.
        returns (np.ndarray): Target returns.

    Returns:
        float: The computed value loss (scalar).
    """
    # Hint:
    # 1. This function is just mean squared error.
    # 2. Compute the elementwise prediction error values - returns.
    # 3. Square it and take the mean to get one scalar.
    ...

